from gemini_utils import ask_gemini, chat
from sentiment import predict_sentiment

def run_chat():
    print("\nHi! I am your restaurant receptionist chatbot.")
    print("I can help you with your experience at the restaurant.")
    print("Type 'exit' or 'quit' anytime to end the chat.\n")

    # Get valid name
    while True:
        name = input("Please enter your name: ").strip()
        if name:
            break
        print("Name cannot be empty. Please try again.")

    # Get valid restaurant name
    while True:
        restaurant_name = input("Please enter the restaurant name: ").strip()
        if restaurant_name:
            break
        print("Restaurant name cannot be empty. Please try again.")

    # Get initial user input
    while True:
        user_input = input("\nHow's your experience? ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Chatbot: Stay safe. Goodbye!")
            return  # exit the whole function

        if user_input:
            break
        print("Input cannot be empty.")

    # Predict sentiment and get Gemini response
    sentiment = predict_sentiment(user_input)
    response = ask_gemini(chat, name, restaurant_name, user_input, sentiment)
    print(f"\nChatbot: {response}")

    # Multi-turn follow-up
    while True:
        follow_up = input("\nYou: ").strip()
        if follow_up.lower() in ["exit", "quit"]:
            print("Chatbot: Thank you, have a great day!")
            break

        response = chat.send_message(follow_up)
        print(f"\nChatbot: {response.text}")

if __name__ == "__main__":
    run_chat()
