import os
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


# Load API key from .env
load_dotenv()

# Gemini embedding model
embedding_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

documents=[
    Document(page_content="Langchain helps developers build LLM applications easily."),
    Document(page_content="Chroma is a vector database optimized for LLM based Search."),
    Document(page_content="Embeddings convert text into high-dimensional vectors."),
    Document(page_content="OpenAI provides powerful embeddings model."),
]

# Create vector store
vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    collection_name="my_collection"
)


# Create retriever
retriever = vector_store.as_retriever(
    search_kwargs={"k": 2}
)

# Test the retriever
query = "What is chroma used for ?"

results = retriever.invoke(query)

print("Number of documents:", len(results))

for i, doc in enumerate(results):
    print(f"\n--- Result {i + 1} ---")
    print(doc.page_content[:2000])