# severity_model.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("./best_sentiment_modelv4")
model = AutoModelForSequenceClassification.from_pretrained("./best_sentiment_modelv4")
model.eval()

label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}


def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    pred = torch.argmax(probs, dim=1).item()
    return label_map[pred]



