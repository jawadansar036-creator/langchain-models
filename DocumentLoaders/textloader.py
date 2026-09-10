from langchain_community.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)

prompt = PromptTemplate(
    template='Write the summary of the following poem-\n {poem}',
    input_variables=['poem']
)

parser = StrOutputParser()

loader = TextLoader('Chatmodels\DocumentLoaders\cricket.txt',encoding='utf-8')

docs = loader.load()

print(type(docs))

print(docs[0].page_content)

print(docs[0].metadata)

chain = prompt | model | parser

result=chain.invoke({'poem':docs[0].page_content})

print(result)