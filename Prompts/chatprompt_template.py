from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)

chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful {domain} expert'),
    ('human', 'Explain in simple terms, what is {topic}')
])


prompt = chat_template.invoke({
    'domain': 'cricket',
    'topic': 'Dusra'
})


response = model.invoke(prompt)
print("AI:", response.content)