import os
import torch
from spam_detection import predict_spam
from toxic_detection import is_toxic
from Review_log import save_review_to_csv
from analyze_review import analyze_review
from sentiment import predict_sentiment
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import streamlit as st
import google.generativeai as genai
from gemini_utils import ask_gemini

os.environ["STREAMLIT_FILE_WATCHER_TYPE"] = "none"

# Load sentiment model
tokenizer = AutoTokenizer.from_pretrained("./best_sentiment_modelv4")
sysmodel = AutoModelForSequenceClassification.from_pretrained("./best_sentiment_modelv4")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sysmodel = sysmodel.to(device)

# Load Gemini
genai.configure(api_key="AIzaSyAdxpHn-sroNiZypmu4Lpbm40JO6zhphSM")
chat_model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    system_instruction="You are a helpful restaurant receptionist chatbot."
)

# Session state initialization
if "chat" not in st.session_state:
    st.session_state.chat = chat_model.start_chat(history=[])
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# UI
st.title("🍽️ Restaurant Receptionist Chatbot")
st.header("We are interested to hear your feedback!")

# Input form
with st.form("Feedback Form"):
    st.subheader("Please fill out the following details:")
    name = st.text_input("What is your name?")
    restaurant_name = st.text_input("What is the restaurant name?")
    user_input = st.text_area("How's your experience?")
    submitted = st.form_submit_button("Submit")

# Handle form submission
if submitted:
    if not name or not restaurant_name or not user_input:
        st.warning("Please fill in all fields.")
    
    elif predict_spam(user_input):
        st.error("Your message has been flagged as spam.")
    elif is_toxic(user_input):
        st.error("Your message contains toxic content.")
    else:
        review = analyze_review(user_input, tokenizer, sysmodel, device)
        save_review_to_csv(review)
        sentiment = review['sentiment']
        response = ask_gemini(st.session_state.chat, name, restaurant_name, user_input, sentiment)
        
        # Show LIME explanation
        lime_exp = review.get("lime", [])
        if lime_exp:
            st.subheader("🔍 LIME Explanation (Top Words Affecting Prediction)")
            for word, score in lime_exp:
                st.write(f"- **{word}** → {score:.3f}")
                
                
        st.session_state.chat_history.append(("user", user_input))
        st.session_state.chat_history.append(("assistant", response))

# Display chat history
for role, msg in st.session_state.chat_history:
    st.chat_message(role).write(msg)

# Multi-turn chat input
follow_up = st.chat_input("Your message:")
if follow_up:
    st.chat_message("user").write(follow_up)
    response = st.session_state.chat.send_message(follow_up).text
    st.chat_message("assistant").write(response)
    st.session_state.chat_history.append(("user", follow_up))
    st.session_state.chat_history.append(("assistant", response))
