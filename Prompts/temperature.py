from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()
model=ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite" ,temperature=1.5)
response=model.invoke("write a 5 line poem on cricket")
print(response.content[0]['text'])