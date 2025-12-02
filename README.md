# Machine Learning for Business Applications (MLBA) Client Retention Model (CRM) 

We introduce a novel hybrid model that combines a RetrievalAugmented Generation (RAG) pipeline with a BERT model to extract a ”Relationship Health Score” from client meeting transcripts. This score, integrated with traditional structured data, is used to train an XGBoost classifier. Key findings demonstrate that our proposed solution achieved an F1-Score of 0.75 and an AUC-ROC of 0.88, significantly outperforming a baseline model that relies only on structured data. Most notably, the model increased prediction recall by 27 percent , enhancing the ability to proactively identify at-risk clients and enabling timely intervention to reduce churn.

## Setup:
* Create a venv with (crm) in python:
```
python -m venv crm
```
* Now activate the virtual environment:
```
source crm/bin/activate
```

* Run below command to install all the necessary dependencies in the virtual environment:
```
pip install -r requirements.txt
```


## Dataset Creation:
* Follow the instructions from the mentioned data source [Dataset](https://github.com/microsoft/topic_conversation/) and copy all the json files generated from the source to the [raw data](./data/source) folder of this project.
* Run the [preprocess_tcr_data.py](./preprocess_tcr_data.py) file to create the final transcripts from the raw data for each meeting in a separate file. 


## BERT MODEL SETUP:
* Run [bert_model.py](./bert_model.py) file. 
```
python3 bert_model.py
```
This will create a folder like [bert-base-uncased](./bert-base-uncased). This folder contains the weights and other model information in order to run in offline mode. 

NOTE: Currently the model weights are not pushed due to large file storage but running the above command will create it. Added a sample directory without the safetensors of the model. 


## Run RAG BERT XGBoost Pipeline:

* Run the pipeline with synthetic to test the pipeline working status:
```
python3 rag_bert_xgboost_pipeline.py
```

* Run the pipeline with real dataset i.e., TCR dataset mentioned above:
```
python3 rag_bert_xgboost_pipeline.py --use_real_dataset
```

Flow of the pipeline:

* The pipeline processes text data through a structured workflow involving preprocessing, model training, and evaluation.

* Raw data is cleaned, tokenised, and converted into model-ready inputs.

* A BERT-based classifier is fine-tuned using Apple’s MPS backend for accelerated training on the M2 Air.

* Multiple training configurations, including ablation variants, are supported to measure the impact of different components.

* Each model is trained and validated across folds, with performance metrics automatically logged.

* The pipeline generates predictions, evaluation reports, and visualisations.

* The framework enables clear comparison between baseline, ablation, and fully optimised models in a reproducible manner.