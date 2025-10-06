from transformers import BertTokenizer, BertModel

model_name = "bert-base-uncased"  # you can change this if you want

# Download model and tokenizer
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# Save them locally
save_path = "./bert-base-uncased"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print("✅ Model and tokenizer saved to", save_path)
