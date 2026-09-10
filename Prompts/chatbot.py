import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Load the API key from .env
load_dotenv()

# Initialize the model
model = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

# Keep track of conversation history
chat_history = [
    SystemMessage(content="You are a helpful, friendly assistant.")
]

print("Chatbot ready! Type 'exit' to exit.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    chat_history.append(HumanMessage(content=user_input))
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))

    print("AI:",result.content[0]['text'])