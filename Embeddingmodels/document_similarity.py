from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

documents ={ "Virat Kohli is a famous Indian cricketer.",
           "Babar Azam is a talented Pakistani batsman.",
           "Steve Smith is a great Australian cricketer.",
           "Ben Stokes is a famous English all-rounder.",
           "Kane Williamson is a skilled New Zealand batsman."
}

query='tell me about Babar Azam'

doc_embeddings = embedding.embed_documents(documents)
query_embdding = embedding.embed_query(query)

print(cosine_similarity([query_embdding], doc_embeddings))

