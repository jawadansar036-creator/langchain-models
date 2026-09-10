import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Load the API key from .env
load_dotenv()

# Initialize the model
model = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

messages = [
    SystemMessage(content='You are a helpful assistant.'),
    HumanMessage(content='Tell me about langchain.')
]
 
result = model.invoke(messages)

messages.append(AIMessage(content=result.content))

print(messages)
 