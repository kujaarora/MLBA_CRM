#!/usr/bin/env python3
"""
RAG -> BERT -> XGBoost full pipeline script.

Replace data loading sections to point to your real TCR dataset and structured features.
"""

import os
import random
import json
import argparse
from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd
from tqdm import tqdm

# NLP
import spacy
from sentence_transformers import SentenceTransformer
import faiss

# Hugging Face
import torch
from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

# ML
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
)
import xgboost as xgb

# -------------------------
# Config
# -------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Model names
SENT_EMB_MODEL = "all-MiniLM-L6-v2"  # sentence-transformers for retrieval (fast)
BERT_MODEL = "bert-base-uncased"  # base model to fine-tune for relationship score

# RAG / retrieval
EMB_DIM = 384  # for all-MiniLM-L6-v2
CHUNK_MAX_TOKENS = 200  # chunk size (approx)

# Relationship score modeling
REL_LABEL_TYPE = "regression"  # options: "regression" or "classification"
# If classification, you must provide label classes (e.g., 3 classes for low/medium/high relationship health)
use_synthetic_dataset = False

# XGBoost
XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "use_label_encoder": False,
    "random_state": SEED,
    "scale_pos_weight": 1.0,  # adjust according to class imbalance
    "base_score": 0.5,
}
N_ROLLING_SPLITS = 5

# Paths
ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# -------------------------
# Utilities: Preprocessing
# -------------------------
nlp = spacy.load("en_core_web_sm", disable=["ner"])  # small spaCy model

from sklearn.utils.class_weight import compute_class_weight

from transformers import AutoModel
import torch.nn as nn
import torch

# Check if the Metal Performance Shaders (MPS) device is available
if torch.backends.mps.is_available():
    # If MPS is available, define the device as 'mps'
    DEVICE = torch.device("mps")
    print("MPS (Apple Silicon GPU) is available. Using the GPU.")
else:
    # Fallback to CPU if MPS is not available
    DEVICE = torch.device("cpu")
    print("MPS is not available. Using the CPU.")


import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score


import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    accuracy_score,
)


import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    accuracy_score,
)


import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    accuracy_score,
)


def compute_metrics(eval_pred):
    logits, labels = eval_pred

    # 1. Handle potential tuple output
    if isinstance(logits, tuple):
        logits = logits[0]

    # 2. Determine Predictions and Probabilities based on shape
    # Case A: Binary Classification (1 output node) - Shape (Batch, 1) or (Batch,)
    if logits.ndim == 1 or (logits.ndim == 2 and logits.shape[1] == 1):
        # Flatten to 1D
        logits = logits.reshape(-1)
        # Sigmoid
        probs = 1 / (1 + np.exp(-logits))
        # Threshold to get Integer Preds (0 or 1)
        preds = (probs > 0.5).astype(int)
        pos_probs = probs

    # Case B: Standard Classification (2+ output nodes) - Shape (Batch, Num_Labels)
    else:
        # Argmax to get Integer Preds (0, 1, 2...)
        preds = np.argmax(logits, axis=-1)
        # Softmax for probabilities
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        # Get probability of the "positive" class (index 1)
        pos_probs = probs[:, 1]

    # 3. SAFETY STEP: Force Labels to Integers and Flatten
    # This prevents the "mix of continuous and binary" error if labels are floats
    labels = labels.astype(int).reshape(-1)
    preds = preds.astype(int).reshape(-1)

    # 4. Compute Metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )

    acc = accuracy_score(labels, preds)

    try:
        roc_auc = roc_auc_score(labels, pos_probs)
    except Exception:
        roc_auc = 0.0

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
    }


class WeightedBERTWithFeatures(nn.Module):
    def __init__(
        self, model_name, handcrafted_dim, hidden_dim=256, num_labels=1, pos_weight=None
    ):
        super().__init__()

        # Load BERT encoder (not classification head)
        self.bert = AutoModel.from_pretrained(model_name, num_labels=num_labels)

        # Combined feature dimension = CLS(768) + handcrafted(K)
        self.input_dim = 768 + handcrafted_dim

        # Small feed-forward classifier on top of concatenated features
        self.classifier = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_labels),
        )

        # Weighted Loss
        if num_labels == 1:
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            self.loss_fn = nn.CrossEntropyLoss(weight=pos_weight)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        handcrafted_features=None,
        labels=None,
        token_type_ids=None,
    ):

        # ---- BERT Encoder ----
        bert_out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # CLS embedding → shape: (batch, 768)
        cls_embed = bert_out.last_hidden_state[:, 0, :]

        # ---- Concatenate handcrafted features ----
        # handcrafted_features shape must be (batch, K)
        if handcrafted_features is None:
            raise ValueError("handcrafted_features must be provided.")

        x = torch.cat([cls_embed, handcrafted_features], dim=1)

        # ---- Classifier ----
        logits = self.classifier(x)

        # ---- Loss ----
        if labels is not None:
            labels = labels.float().unsqueeze(1)  # BCE expects (N,1)
            loss = self.loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}

        return {"logits": logits}


from textblob import TextBlob


def get_responsiveness_facet(transcript_text):
    """Calculates ratio of Client words vs Total words."""
    client_word_count = 0
    total_word_count = 0

    lines = transcript_text.split("\n")
    for line in lines:
        words = len(line.split())
        total_word_count += words
        if line.lower().startswith("client:"):
            client_word_count += words

    if total_word_count == 0:
        return 0.0
    return client_word_count / total_word_count


def get_interruption_probe(transcript_text):
    """
    Counts how many times speakers switch rapidly (turns with < 5 words).
    High count = Heated debate or high engagement.
    """
    lines = transcript_text.split("\n")
    short_turns = 0

    for line in lines:
        # Remove "Client:" or "Agent:" prefix before counting
        content = line.split(":", 1)[-1] if ":" in line else line
        if len(content.split()) < 5:
            short_turns += 1

    # Normalize by conversation length
    return short_turns / len(lines) if lines else 0


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def get_topic_shift_probe(transcript_text):
    # Split conversation in half
    mid = len(transcript_text) // 2
    part1 = transcript_text[:mid]
    part2 = transcript_text[mid:]

    if not part1 or not part2:
        return 0.0

    # Vectorize and compare
    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform([part1, part2])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        return similarity[0][0]  # 1.0 = same topic, 0.0 = total shift
    except:
        return 0.0


RISK_KEYWORDS = [
    "cancel",
    "expensive",
    "competitor",
    "switch",
    "unhappy",
    "problem",
    "issue",
    "refund",
    "manager",
    "disappointed",
    "terminate",
]


def get_risk_term_facet(transcript_text):
    """Counts occurrences of risk words."""
    count = 0
    lower_text = transcript_text.lower()
    for word in RISK_KEYWORDS:
        count += lower_text.count(word)
    return count


# Or use 'vaderSentiment' for better social media/chat handling
def get_sentiment_facet(transcript_text):
    """Returns a score from -1 (Negative) to 1 (Positive)"""
    blob = TextBlob(transcript_text)
    return blob.sentiment.polarity


import re


def extract_handcrafted_features(transcript_text):
    """
    Extracts a 5-dimensional vector of interpretable conversational features from a transcript.

    Returns:
        np.array: [Sentiment, Responsiveness, Risk_Count, Interruption_Rate, Topic_Coherence]
    """
    if not transcript_text or not isinstance(transcript_text, str):
        # Return a zero vector if text is empty or invalid
        return np.zeros(5)

    # --- 1. Sentiment Score (Emotional Tone) ---
    # Range: -1.0 (Negative) to 1.0 (Positive)
    try:
        sentiment_score = get_sentiment_facet(transcript_text)
    except:
        sentiment_score = 0.0

    # --- 2. Responsiveness (Client Talk Ratio) ---
    # Ratio of words spoken by "Client" vs Total words.
    # Assumes "Client:" or "Customer:" prefixes. If not found, defaults to 0.5.
    responsiveness = get_responsiveness_facet(transcript_text)

    total_words = 0
    for line in transcript_text.split("\n"):
        words = len(line.split())
        total_words += words

    # --- 3. Risk Term Density (Risk Keywords) ---
    # Count of risk words normalized by transcript length (per 100 words)
    text_lower = transcript_text.lower()
    risk_count = get_risk_term_facet(text_lower)

    # Normalize: Risk words per 100 words (to handle varying transcript lengths)
    # Adding 1 to total_words to avoid division by zero
    risk_density = (risk_count / (total_words + 1)) * 100

    # --- 4. Interruption Pattern (Short Turns) ---
    # Fraction of turns that are very short (< 5 words), indicating rapid back-and-forth
    short_turns = 0
    valid_lines = [
        line for line in transcript_text.split("\n") if line.strip()
    ]  # Filter empty lines

    for line in valid_lines:
        # Strip speaker labels (e.g. "Agent: ") to count actual content
        content = line.split(":", 1)[-1] if ":" in line else line
        if len(content.split()) < 5:
            short_turns += 1

    interruption_rate = short_turns / len(valid_lines) if valid_lines else 0.0

    # --- 5. Topic Coherence (Shift between 1st and 2nd half) ---
    # Cosine similarity between the first half and second half of the text.
    # 1.0 = Very consistent topic, 0.0 = Complete topic shift
    try:
        topic_coherence = get_topic_shift_probe(transcript_text)
    except:
        topic_coherence = 0.0

    # --- Combine into Vector ---
    feature_vector = np.array(
        [
            sentiment_score,
            responsiveness,
            risk_density,
            interruption_rate,
            topic_coherence,
        ]
    )

    return feature_vector


def preprocess_text(text: str) -> str:
    """
    Clean, tokenise, lemmatise using spaCy. Keep sentence boundaries.
    """
    if not isinstance(text, str):
        return ""
    # Basic cleaning
    text = text.replace("\n", " ").strip()
    doc = nlp(text)
    tokens = []
    for sent in doc.sents:
        sent_tok = []
        for token in sent:
            if token.is_space or token.is_punct:
                continue
            # Keep alpha-numeric tokens, lemmatise, lower
            lemma = token.lemma_.lower()
            if lemma == "-pron-":
                lemma = token.text.lower()
            sent_tok.append(lemma)
        if sent_tok:
            tokens.append(" ".join(sent_tok))
    return " . ".join(tokens)  # preserve sentence separators


def chunk_text(text: str, max_tokens: int = CHUNK_MAX_TOKENS) -> List[str]:
    """
    Naive chunking by words. Better: chunk by sentences until approximate token limit reached.
    """
    if not text:
        return []
    words = text.split()
    chunks = []
    cur = []
    for w in words:
        cur.append(w)
        if len(cur) >= max_tokens:
            chunks.append(" ".join(cur))
            cur = []
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def get_raw_file_contents_dataset(real_dataset_path):
    raw_data_list = []
    for root, _, files in os.walk(real_dataset_path):
        for file in files:
            if file.endswith(".json"):
                json_file_path = os.path.join(root, file)
                with open(json_file_path, "r") as file:
                    dataset = json.loads(file.read())
                    raw_data_list.append(dataset)
                    print("Appending the client data: ", json_file_path)
    return raw_data_list


def load_real_dataset(real_dataset_path):

    # 2. Pre-process the raw JSON into usable transcript strings
    # The JSON structure is {'Topic': ["'SPEAKER': 'Text'", ...]}
    # We clean this to "SPEAKER: Text" and join into one string per topic.
    raw_data_list = get_raw_file_contents_dataset(real_dataset_path=real_dataset_path)
    clients_data_dict = {}

    # 2. Parse JSON into a list of full meeting transcripts
    # We iterate through the raw data to construct complete text blocks for each meeting.
    for i in range(len(raw_data_list)):
        real_transcripts_pool = []

        # Iterate over meetings (e.g., "en_dev_004...")
        print(raw_data_list[i].keys())
        doc_key = list(raw_data_list[i].keys())[0]
        for meeting_id, meeting_data in raw_data_list[i][doc_key].items():
            full_meeting_text = []

            # Sort topics by start time to maintain chronological flow
            topics = meeting_data.get("topics", {})
            sorted_topics = sorted(
                topics.items(), key=lambda item: item[1].get("topic_start_s", 0.0)
            )

            for topic_name, topic_data in sorted_topics:
                # Add a header for the topic (optional, but good for context)
                full_meeting_text.append(f"\n[Topic: {topic_name}]")

                # Iterate through transcripts in this topic
                for segment in topic_data.get("transcripts", []):
                    speaker = segment.get("speaker", "Unknown")
                    content = segment.get("contents", "")
                    full_meeting_text.append(f"{speaker}: {content}")

            # Join all lines to form one document
            real_transcripts_pool.append("\n".join(full_meeting_text))

        clients_data_dict[f"client_{i}"] = real_transcripts_pool
    # 3. Generate DataFrames
    # We will assign the parsed real transcripts to the generated clients.

    rng = np.random.RandomState(SEED)

    transcripts_rows = []
    structured_rows = []

    industries = ["SaaS", "Finance", "Retail", "Health", "Tech"]

    for i in range(len(raw_data_list)):

        # --- Structured Data Generation (Synthetic) ---
        tenure = rng.randint(1, 60)
        contract_size = float(rng.randint(1, 100)) * 1000.0
        industry = rng.choice(industries)
        other_numeric = rng.rand()
        churn = int(rng.rand() < 0.15)  # 15% churn baseline

        structured_rows.append(
            {
                "client_id": f"client_{i}",
                "date": pd.Timestamp("2023-01-01")
                + pd.Timedelta(days=int(rng.randint(0, 300))),
                "tenure_months": tenure,
                "contract_size": contract_size,
                "industry": industry,
                "other_numeric": other_numeric,
                "churn": churn,
            }
        )

        # --- Transcript Data Generation (Real Text) ---
        meeting_id = f"client_{i}_m{1}"

        # Select a random transcript from our parsed real pool
        # Since we only have a few real meetings, we sample with replacement
        transcript_text = rng.choice(real_transcripts_pool)

        meeting_date = pd.Timestamp("2022-01-01") + pd.Timedelta(
            days=int(rng.randint(0, 600))
        )

        transcripts_rows.append(
            {
                "client_id": f"client_{i}",
                "meeting_id": meeting_id,
                "transcript": transcript_text,
                "meeting_date": meeting_date,
            }
        )

    transcripts_df = pd.DataFrame(transcripts_rows)
    structured_df = pd.DataFrame(structured_rows)

    return transcripts_df, structured_df


# -------------------------
# Data loading (placeholder / synthetic example)
# -------------------------
def load_synthetic_dataset(n_clients=2000) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      transcripts_df: columns = ['client_id', 'meeting_id', 'transcript', 'meeting_date']
      structured_df: columns = ['client_id', 'date', 'tenure_months', 'contract_size', 'industry', 'other_numeric', 'churn']
    This generates synthetic data for demonstration. Replace with actual loading code.
    """
    rng = np.random.RandomState(SEED)
    clients = [f"client_{i}" for i in range(n_clients)]
    transcripts_rows = []
    structured_rows = []
    industries = ["SaaS", "Finance", "Retail", "Health"]
    for c in clients:
        n_meetings = rng.randint(1, 6)
        tenure = rng.randint(1, 60)
        contract_size = float(rng.randint(1, 100)) * 1000.0
        industry = rng.choice(industries)
        other_numeric = rng.rand()
        churn = int(rng.rand() < 0.15)  # 15% churn baseline
        # create structured time series: we'll put a single row per client for simplicity, with a 'date'
        structured_rows.append(
            {
                "client_id": c,
                "date": pd.Timestamp("2023-01-01")
                + pd.Timedelta(days=int(rng.randint(0, 300))),
                "tenure_months": tenure,
                "contract_size": contract_size,
                "industry": industry,
                "other_numeric": other_numeric,
                "churn": churn,
            }
        )
        # meetings
        for m in range(n_meetings):
            meeting_id = f"{c}_m{m}"
            # generate synthetic transcript text
            words = [
                "project",
                "deadline",
                "budget",
                "thanks",
                "concern",
                "happy",
                "dissatisfied",
                "upgrade",
                "contract",
                "cancel",
            ]
            transcript = " ".join(rng.choice(words, size=rng.randint(30, 300)))
            meeting_date = pd.Timestamp("2022-01-01") + pd.Timedelta(
                days=int(rng.randint(0, 600))
            )
            transcripts_rows.append(
                {
                    "client_id": c,
                    "meeting_id": meeting_id,
                    "transcript": transcript,
                    "meeting_date": meeting_date,
                }
            )
    transcripts_df = pd.DataFrame(transcripts_rows)
    structured_df = pd.DataFrame(structured_rows)
    return transcripts_df, structured_df


# -------------------------
# Build retrieval index (RAG retrieval using SBERT + FAISS)
# -------------------------
class RAGRetriever:
    def __init__(self, model_name: str = SENT_EMB_MODEL, emb_dim: int = EMB_DIM):
        self.model = SentenceTransformer(model_name)
        self.emb_dim = emb_dim
        self.index = None
        self.metadata = (
            []
        )  # maps index -> {client_id, meeting_id, chunk_text, meeting_date, ...}

    def build_index(
        self, transcripts_df: pd.DataFrame, chunk_max_tokens=CHUNK_MAX_TOKENS
    ):
        """
        Preprocess transcripts -> chunk -> embed -> faiss index
        transcripts_df: has client_id, meeting_id, transcript, meeting_date
        """
        print("[RAG] Preprocessing and chunking transcripts...")
        corpus_chunks = []
        meta = []
        for _, row in tqdm(transcripts_df.iterrows(), total=len(transcripts_df)):
            client_id = row["client_id"]
            meeting_id = row["meeting_id"]
            raw = preprocess_text(row["transcript"])
            chunks = chunk_text(raw, max_tokens=chunk_max_tokens)
            for c in chunks:
                meta.append(
                    {
                        "client_id": client_id,
                        "meeting_id": meeting_id,
                        "chunk_text": c,
                        "meeting_date": row.get("meeting_date", pd.NaT),
                    }
                )
                corpus_chunks.append(c)
        print(
            f"[RAG] Embedding {len(corpus_chunks)} chunks with {self.model.__class__.__name__}..."
        )
        if len(corpus_chunks) == 0:
            raise ValueError("No chunks found; check transcripts.")
        embeddings = self.model.encode(
            corpus_chunks, show_progress_bar=True, convert_to_numpy=True
        )
        # build FAISS index
        index = faiss.IndexFlatIP(
            self.emb_dim
        )  # inner product; ensure vectors normalized
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        self.index = index
        self.metadata = meta
        self.corpus = corpus_chunks
        print(f"[RAG] Index built with {index.ntotal} vectors.")

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        q_emb = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        D, I = self.index.search(q_emb, top_k)
        results = []
        for dist, idx in zip(D[0], I[0]):
            if idx < 0:
                continue
            meta = self.metadata[idx].copy()
            meta["score"] = float(dist)
            results.append(meta)
        return results

    def retrieve_for_client(
        self, client_id: str, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Simple: use client aggregated meetings as query (or pick latest meeting) — here we build a small query
        from their latest transcript or aggregate of structured features. For now: use all chunks for that client
        and select highest scoring when querying with an aggregated prompt. This function is a placeholder —
        in practice you'd craft queries per client/timepoint.
        """
        # Create query as placeholder:
        query = f"client {client_id} relationship meeting"
        return self.retrieve(query, top_k=top_k)


# -------------------------
# Relationship Health Model (BERT-based)
# -------------------------
def prepare_rel_dataset(client_snippets: pd.DataFrame, labels: pd.Series):
    """
    Build HF dataset for fine-tuning BERT on snippet(s) -> relationship health.
    client_snippets: DataFrame with columns ['client_id', 'snippet'] where snippet is concatenated retrieved snippets per client
    labels: Series aligned with client_snippets index of numeric score or class
    """
    df = client_snippets.copy()
    df["label"] = labels.values
    # Use HuggingFace Dataset for convenience
    ds = Dataset.from_pandas(df[["snippet", "label"]])
    return ds


def tokenize_function(examples, tokenizer):
    return tokenizer(examples["snippet"], truncation=True, max_length=512)


def fine_tune_relation_model(
    train_ds,
    val_ds,
    model_name=BERT_MODEL,
    label_type="regression",
    out_dir="artifacts/relationship_model",
):
    """
    Train BERT to predict relationship health from retrieved snippets.
    label_type: "regression" or "classification"
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def process_data(examples):
        # 1. Tokenize (existing logic)
        tokenized_inputs = tokenize_function(examples, tokenizer)

        # 2. Extract handcrafted features
        # Assuming your input text column is named "text". Adjust if it's "content", "sentence", etc.
        # We iterate because 'examples["text"]' is a list when batched=True
        features = [extract_handcrafted_features(text) for text in examples["snippet"]]

        # Add the new key expected by your model's forward() method
        tokenized_inputs["handcrafted_features"] = features
        return tokenized_inputs

    if label_type == "regression":
        pos_weight = torch.tensor([3.5]).to(DEVICE)
        model = WeightedBERTWithFeatures(
            model_name="/Users/yash3886/Desktop/GIM_PROJECT/MLBA_GIM_CSM/bert-base-uncased",
            num_labels=1,
            handcrafted_dim=5,
            pos_weight=pos_weight,
        )

    else:
        # classification - determine number of classes from labels in dataset
        unique_labels = sorted(set(train_ds["label"]))
        num_labels = len(unique_labels)
        model = AutoModel.from_pretrained(model_name, num_labels=num_labels)

    tokenized_train = train_ds.map(
        process_data,
        batched=True,
        remove_columns=["snippet"],  # <--- CRITICAL FIX: Remove raw text
    )

    tokenized_val = val_ds.map(
        process_data,
        batched=True,
        remove_columns=["snippet"],  # <--- CRITICAL FIX: Remove raw text
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=out_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=8,
        num_train_epochs=100,
        weight_decay=0.01,
        warmup_ratio=0.1,
        load_best_model_at_end=True,
        metric_for_best_model=(
            "recall" if label_type == "regression" else "eval_accuracy"
        ),
        logging_steps=50,
        fp16=torch.backends.mps.is_available(),
        seed=SEED,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    return out_dir  # path to saved model


def predict_relation_score(
    retriever: RAGRetriever,
    bert_model_dir: str,
    clients: List[str],
    top_k=5,
    label_type="regression",
):
    """
    For each client, retrieve snippets, build a combined snippet text, feed to fine-tuned BERT model, and return score.
    """
    tokenizer = AutoTokenizer.from_pretrained(bert_model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(bert_model_dir)
    model.eval()
    model.to(DEVICE)

    client_scores = {}
    for c in tqdm(clients):
        retrieved = retriever.retrieve_for_client(c, top_k=top_k)
        # combine retrieved chunks into single text (ordered by score)
        combined = " ".join([r["chunk_text"] for r in retrieved])
        # tokenize and predict
        inputs = tokenizer(
            combined, truncation=True, max_length=512, return_tensors="pt"
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs)
            logits = out.logits.cpu().numpy()
            if label_type == "regression":
                score = float(logits.squeeze())  # direct regression logit
            else:
                # classification: convert to probability of positive / map to numeric
                probs = torch.softmax(out.logits, dim=-1).cpu().numpy()
                # assume higher class indices -> better health; compute expected class
                classes = np.arange(probs.shape[-1])
                score = float((probs * classes).sum())
        client_scores[c] = score
    return client_scores


# -------------------------
# Merge relationship score with structured features
# -------------------------
def prepare_xgb_dataset(
    structured_df: pd.DataFrame,
    relation_scores: Dict[str, float],
    feature_cols: List[str],
    label_col="churn",
):
    df = structured_df.copy()
    df["relation_score"] = df["client_id"].map(relation_scores).fillna(0.0)
    # one-hot encode industry if present
    if "industry" in df.columns:
        df = pd.get_dummies(df, columns=["industry"], drop_first=True)
    X = df[feature_cols + ["relation_score"]].fillna(0.0)
    y = df[label_col].astype(int)
    dates = (
        df["date"]
        if "date" in df.columns
        else pd.Series(pd.Timestamp("2023-01-01"), index=df.index)
    )
    return X, y, dates


# -------------------------
# Time-based rolling-origin evaluation
# -------------------------
def rolling_origin_evaluation(
    X: pd.DataFrame, y: pd.Series, dates: pd.Series, n_splits=N_ROLLING_SPLITS
):
    """
    Create rolling-origin splits using chronological order of 'dates'. For each split,
    train an XGBoost model (with class weighting via scale_pos_weight) and compute evaluation metrics.
    Returns aggregated metrics across splits.
    """
    df = X.copy()
    df["label"] = y.values
    df["date"] = dates.values
    df.sort_values("date", inplace=True)
    # Create split points
    unique_dates = df["date"].dropna().unique()
    # Use TimeSeriesSplit-like approach but on rows: we will split by index windows
    tss = TimeSeriesSplit(n_splits=n_splits)
    metrics = []
    i = 0
    for train_idx, test_idx in tss.split(df):
        i += 1
        train = df.iloc[train_idx]
        test = df.iloc[test_idx]
        X_train = train.drop(columns=["label", "date"])
        y_train = train["label"]
        X_test = test.drop(columns=["label", "date"])
        y_test = test["label"]

        # compute scale_pos_weight
        pos = (y_train == 1).sum()
        neg = (y_train == 0).sum()
        scale_pos_weight = (neg / pos) if pos > 0 else 1.0
        params = XGB_PARAMS.copy()
        params["scale_pos_weight"] = scale_pos_weight

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=200,
            evals=[(dtrain, "train"), (dtest, "eval")],
            early_stopping_rounds=20,
            verbose_eval=False,
        )
        # predict probabilities
        y_pred_proba = model.predict(dtest)
        y_pred = (y_pred_proba >= 0.5).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="binary", zero_division=0
        )
        roc = (
            roc_auc_score(y_test, y_pred_proba)
            if len(np.unique(y_test)) > 1
            else float("nan")
        )
        pr_auc = (
            average_precision_score(y_test, y_pred_proba)
            if len(np.unique(y_test)) > 1
            else float("nan")
        )
        metrics.append(
            {
                "split": i,
                "precision": p,
                "recall": r,
                "f1": f1,
                "roc_auc": roc,
                "pr_auc": pr_auc,
            }
        )
    metrics_df = pd.DataFrame(metrics)
    return metrics_df


# -------------------------
# Cost / Threshold analysis and policy simulation
# -------------------------
def threshold_cost_analysis(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    cost_tp: float = -100.0,
    cost_fp: float = -10.0,
    cost_fn: float = -1000.0,
    cost_tn: float = 0.0,
):
    """
    Evaluate costs for a range of thresholds. Negative costs indicate spending/loss; positive indicates savings/gain.
    Default numbers are placeholders: set to real business values (e.g., cost of outreach, revenue saved).
    """
    thresholds = np.linspace(0.0, 1.0, 101)
    rows = []
    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        tp = np.sum((preds == 1) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))
        fn = np.sum((preds == 0) & (y_true == 1))
        tn = np.sum((preds == 0) & (y_true == 0))
        total_cost = tp * cost_tp + fp * cost_fp + fn * cost_fn + tn * cost_tn
        rows.append(
            {
                "threshold": t,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "total_cost": total_cost,
            }
        )
    return pd.DataFrame(rows)


def simulate_policy_top_k(
    y_true: np.ndarray, y_proba: np.ndarray, k_percent: float = 0.05
):
    """
    Flag top-k% clients by predicted probability for outreach; compute recall and precision among flagged.
    """
    n = len(y_proba)
    k = max(1, int(n * k_percent))
    idx_sorted = np.argsort(-y_proba)  # descending
    flagged_idx = idx_sorted[:k]
    flagged_labels = y_true[flagged_idx]
    precision = flagged_labels.mean() if len(flagged_labels) > 0 else 0.0
    recall = flagged_labels.sum() / y_true.sum() if y_true.sum() > 0 else 0.0
    return {
        "k_percent": k_percent,
        "k_count": k,
        "precision": precision,
        "recall": recall,
    }


# -------------------------
# Full run demo (synthetic)
# -------------------------
def main_demo(dataset_path=None):
    if use_synthetic_dataset:
        print("Loading synthetic dataset...")
        transcripts_df, structured_df = load_synthetic_dataset(n_clients=1000)
    else:
        print("Loading real dataset...")
        transcripts_df, structured_df = load_real_dataset(dataset_path)

    # Build RAG retrieval index
    retriever = RAGRetriever(model_name=SENT_EMB_MODEL, emb_dim=EMB_DIM)
    retriever.build_index(transcripts_df)

    # Prepare client-level aggregated retrieved snippets and "ground truth" labels (for relation model training)
    # For demo: we'll generate a synthetic relationship_score from structured churn/tenure heuristics to train BERT
    print("Preparing client-level dataset for relationship model...")
    # Build per-client aggregated snippet by concatenating top-k retrievals for that client
    clients = structured_df["client_id"].unique().tolist()
    client_snippets = []
    client_ids = []
    relation_labels = []
    for c in tqdm(clients):
        retrieved = retriever.retrieve_for_client(c, top_k=5)
        combined = " ".join([r["chunk_text"] for r in retrieved]) if retrieved else ""
        client_snippets.append({"client_id": c, "snippet": combined})
        client_ids.append(c)
        # synthetic label: lower tenure & high contract_size -> better health (just example)
        row = structured_df[structured_df["client_id"] == c].iloc[0]
        synthetic_score = (
            1.0
            + 0.01 * row["tenure_months"]
            - 0.000001 * row["contract_size"]
            + row["other_numeric"]
        )
        # Normalize to 0-1
        relation_labels.append(float(np.clip((synthetic_score - 0.5) / 2.0, 0.0, 1.0)))

    client_snippets_df = pd.DataFrame(client_snippets).set_index("client_id")
    relation_labels_arr = np.array(relation_labels)

    # Split for relation model training
    train_idx, val_idx = train_test_split(
        np.arange(len(client_ids)), test_size=0.2, random_state=SEED
    )
    train_df = client_snippets_df.iloc[train_idx].reset_index()
    val_df = client_snippets_df.iloc[val_idx].reset_index()
    train_labels = relation_labels_arr[train_idx]
    val_labels = relation_labels_arr[val_idx]

    train_ds = prepare_rel_dataset(
        train_df.rename(columns={"snippet": "snippet"}), pd.Series(train_labels)
    )
    val_ds = prepare_rel_dataset(
        val_df.rename(columns={"snippet": "snippet"}), pd.Series(val_labels)
    )

    print("Fine-tuning Relationship (BERT) model (this may take a while)...")
    rel_model_dir = fine_tune_relation_model(
        train_ds,
        val_ds,
        model_name=BERT_MODEL,
        label_type=REL_LABEL_TYPE,
        out_dir=os.path.join(ARTIFACT_DIR, "rel_model"),
    )

    # Predict relationship scores for all clients with the trained model
    print("Predicting Relationship Health Scores for all clients...")
    client_scores = predict_relation_score(
        retriever, rel_model_dir, clients, top_k=5, label_type=REL_LABEL_TYPE
    )

    # Prepare data for XGBoost
    feature_cols = ["tenure_months", "contract_size", "other_numeric"]
    # ensure industry one-hot included, get_dummies will add the columns
    X, y, dates = prepare_xgb_dataset(
        structured_df, client_scores, feature_cols, label_col="churn"
    )

    # Split for demo: last 20% as test chronologically
    data_df = X.copy()
    data_df["churn"] = y.values
    data_df["date"] = dates.values
    data_df.sort_values("date", inplace=True)
    split_idx = int(len(data_df) * 0.8)
    train_df = data_df.iloc[:split_idx]
    test_df = data_df.iloc[split_idx:]

    X_train = train_df.drop(columns=["churn", "date"])
    y_train = train_df["churn"]
    X_test = test_df.drop(columns=["churn", "date"])
    y_test = test_df["churn"]

    # XGBoost train
    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    scale_pos_weight = (neg / pos) if pos > 0 else 1.0
    params = XGB_PARAMS.copy()
    params["scale_pos_weight"] = scale_pos_weight

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    print("Training XGBoost classifier...")
    xgb_model = xgb.train(
        params,
        dtrain,
        num_boost_round=300,
        evals=[(dtrain, "train"), (dtest, "eval")],
        early_stopping_rounds=20,
        verbose_eval=10,
    )

    y_proba = xgb_model.predict(dtest)
    y_pred = (y_proba >= 0.5).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )
    roc = roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else float("nan")
    pr_auc = (
        average_precision_score(y_test, y_proba)
        if len(np.unique(y_test)) > 1
        else float("nan")
    )
    print("=== XGBoost Test Metrics ===")
    print(
        f"Precision: {p:.4f}  Recall: {r:.4f}  F1: {f1:.4f}  ROC-AUC: {roc:.4f}  PR-AUC: {pr_auc:.4f}"
    )

    # Rolling-origin eval across the whole dataset
    print("Performing rolling-origin evaluation across dataset...")
    metrics_df = rolling_origin_evaluation(X, y, dates, n_splits=5)
    print(metrics_df)
    metrics_df.to_csv(os.path.join(ARTIFACT_DIR, "rolling_metrics.csv"), index=False)

    # Cost / threshold analysis on test set
    print("Running threshold cost analysis on test set...")
    cost_df = threshold_cost_analysis(
        y_test.values,
        y_proba,
        cost_tp=-50.0,
        cost_fp=-10.0,
        cost_fn=-1000.0,
        cost_tn=0.0,
    )
    best = cost_df.loc[cost_df["total_cost"].idxmin()]
    print("Best threshold by cost:", best.to_dict())
    cost_df.to_csv(os.path.join(ARTIFACT_DIR, "cost_thresholds.csv"), index=False)

    # Policy simulation: top-k% outreach
    sim = simulate_policy_top_k(y_test.values, y_proba, k_percent=0.05)
    print("Policy simulation for top-5% outreach:", sim)

    # Save artifacts
    xgb_model.save_model(os.path.join(ARTIFACT_DIR, "xgb_model.json"))
    # save relation model already saved
    pd.DataFrame.from_dict(
        client_scores, orient="index", columns=["relation_score"]
    ).to_csv(os.path.join(ARTIFACT_DIR, "relation_scores.csv"))

    print("Done. Artifacts saved to", ARTIFACT_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the RAG/BERT/XGBoost Pipeline")
    dataset_path = "data/source"

    # 'action="store_true"' implies:
    # If the user types --real, the variable becomes True.
    # If the user does NOT type it, the variable defaults to False.
    parser.add_argument(
        "--use_real_dataset",
        action="store_true",
        help="Use the real JSON dataset instead of synthetic data.",
    )

    args = parser.parse_args()

    # Pass the boolean value to your function
    if args.use_real_dataset:
        main_demo(dataset_path)
    else:
        main_demo()
