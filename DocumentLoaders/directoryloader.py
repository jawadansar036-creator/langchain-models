from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)

loader = DirectoryLoader('Chatmodels/DocumentLoaders/books',
    glob='*.pdf',
    loader_cls=PyPDFLoader
    )

# docs = loader.load()

docs = loader.lazy_load()

for document in docs:
    print(document.metadata)

# print(len(docs))

# print(docs[5].page_content)

# print(docs[5].metadata)