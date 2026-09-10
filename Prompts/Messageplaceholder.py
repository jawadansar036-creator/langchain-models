from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pathlib import Path

#chat template
chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful support assistance'),
    MessagesPlaceholder(variable_name= 'chat_history'),
    ('human', '{query}')
])

chat_history = []
# load chat history
file_path = Path(__file__).parent / "chat_history.txt"

with open(file_path, "r", encoding="utf-8") as f:
    chat_history = f.readlines()
    
prompt = chat_template.invoke({'chat_history':chat_history,'query':'Where is my refund'})
print(prompt)