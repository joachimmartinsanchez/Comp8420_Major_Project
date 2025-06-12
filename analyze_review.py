from sentiment import predict_sentiment
import spacy
from lime.lime_text import LimeTextExplainer
import torch

# Load spaCy model once
nlp = spacy.load("en_core_web_sm")

# Full analysis function
def analyze_review(text, tokenizer, model, device):
    print(f"\n🔍 Input: {text}\n")

    # NER
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    print(f"🏷️ Named Entities: {entities if entities else 'None'}")

    # SENTIMENT
    sentiment = predict_sentiment(text)
    print(f"💬 Sentiment: {sentiment.upper()}")

    # LIME
    explainer = LimeTextExplainer(class_names=['Negative', 'Neutral', 'Positive'])

    def predict_proba(texts):
        tokens = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**tokens)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        return probs.cpu().numpy()

    print("📌 LIME Explanation (Top Contributing Words):")
    exp = explainer.explain_instance(text, predict_proba, num_features=2)
    lime_explanation = exp.as_list()
    for word, score in lime_explanation:
        print(f"   - {word}: {score:.3f}")
        
        
    return {
        "text": text,
        "entities": entities,
        "sentiment": sentiment,
        "lime": lime_explanation
    }
