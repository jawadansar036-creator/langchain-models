from langchain_community.document_loaders import CSVLoader

loader = CSVLoader('Chatmodels/DocumentLoaders/data.csv')

docs = loader.load()

print(docs[0].page_content)