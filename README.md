# Comp8420_Major_Project

---
## Files

restaurant_review_notebook.ipynb - Training, evaluation, confusion matrix & model saving
analyze_review.py - NER, Lime and Sentiment Prediction function is stored in this file
Review_log.py - function that saves the customer review, sentiment and NER into .csv file for the management use is inside this file
sentiment.py - predict_sentiment() function using trained DistilBERT
spam_detection.py - spam filter function
toxic_detection.py - toxic comments filter function
gemini_utils.py - ask_gemini() function handling sentiment-based prompts
Chatbot.py - Terminal-based chatbot interface
main.py - Streamlit frontend for running the chatbot
---

## Features

- Filters toxic and spam messages
- saves customer review, sentiment and NER for management review purposes
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

{'eval_loss': 1.0697822570800781,
 'eval_accuracy': 0.8433734939759037,
 'eval_precision': 0.8468368666975595,
 'eval_recall': 0.8433734939759037,
 'eval_f1': 0.844767714337081,
 'eval_runtime': 75.6384,
 'eval_samples_per_second': 13.168,
 'eval_steps_per_second': 0.833,
 'epoch': 10.0}

---

## Chatbot Behavior

- **Positive**: Thank user + suggest dishes + 10% referral discount  
- **Neutral**: Thank user + ask for improvement suggestions  
- **Negative**: Apologize + ask for details + 30% discount with code

---

