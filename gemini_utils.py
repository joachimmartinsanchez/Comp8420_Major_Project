import google.generativeai as genai

genai.configure(api_key="AIzaSyAdxpHn-sroNiZypmu4Lpbm40JO6zhphSM")

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash"
)

chat = model.start_chat(history=[])

def ask_gemini(chat, name, restaurant_name, user_input, sentiment):
    prompt = f"""
    You are a helpful restaurant receptionist chatbot. Your task is to interact with the customer regarding his/her experience at the {restaurant_name}.
    
    greet the customer by "{name}" and thank them for their feedback.
    Provide a summary of the customer's input and the sentiment analysis result.
    The customer's input is as follows:
    
    User input: "{user_input}"
    The system has analyzed the user's input and provided the following predictions: "{sentiment}".
    """
    if sentiment == "positive":
        prompt += f"""thank the customer for their kind words and promote other food that the {restaurant_name} offers. Tell them that they can have 10% discount if they recommend it to their friends and family. If they reply that they will recommend. provide a 10 digit discount code starting with DISC they can use when they order again."""
        
    elif sentiment == "neutral":
        prompt += "\n\nSince the sentiment is neutral, thank the customer for their feedback and ask if there is something you could do to make their experience better."
    else:  # positive sentiment
        prompt += "\n\n Apologize for the inconvenience and ask for more details about their experience. Offer to resolve any issues they faced. Offer a 30% discount with random code to compensate for the bad experience."
    
    
    response = chat.send_message(prompt)
    
    return response.text