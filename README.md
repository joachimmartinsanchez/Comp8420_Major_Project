# Comp8420_Major_Project

---
## Files

restaurant_review.ipynb - Training, evaluation, confusion matrix & model saving
sentiment.py - predict_sentiment() function using trained DistilBERT
gemini_utils.py - ask_gemini() function handling sentiment-based prompts
Chatbot.py - Terminal-based chatbot interface
main.py - Streamlit frontend for running the chatbot
---

## Features

- Classifies sentiment as **Positive**, **Neutral**, or **Negative**
- Chatbot generates LLM-based personalized replies
- Offers automatic discount codes depending on feedback sentiment
- Displays metrics and confusion matrix during testing

---

##  How to Run
Run the chatbot (Terminal version)
python Chatbot.py

Run the Streamlit web interface
streamlit run main.py

---

## Model Performance

| Metric     | Score   |
|------------|---------|
| Accuracy   | 78.76%  |
| Precision  | 78.40%  |
| Recall     | 78.76%  |
| F1 Score   | 78.46%  |

- Strong performance on positive/negative reviews
- Minor confusion with **neutral** class (can be improved with more balanced data)

---

## Chatbot Behavior

- **Positive**: Thank user + suggest dishes + 10% referral discount  
- **Neutral**: Thank user + ask for improvement suggestions  
- **Negative**: Apologize + ask for details + 30% discount with code

---

