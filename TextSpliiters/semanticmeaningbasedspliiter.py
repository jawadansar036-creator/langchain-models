from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model="model=gemini-3.5-flash-lite" 
)


text_splitter = SemanticChunker(
    GoogleGenerativeAIEmbeddings(),breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=1
    )


text = """
Cricket is one of the most popular sports in the world. A batsman tries to score runs while the bowler attempts to take wickets. Teams use different strategies depending on the pitch and weather conditions. A good captain must understand the game and make quick decisions.
Python is a popular programming language used for web development, artificial intelligence, and data analysis. Classes and objects are important concepts in object-oriented programming. Python also provides many libraries that make complex tasks easier.
The weather can have a major impact on daily life. Heavy rain can cause flooding and make roads difficult to travel on. On the other hand, sunny weather is often preferred for outdoor activities. Farmers also depend heavily on weather conditions for their crops.
Artificial intelligence is changing the way people interact with technology. Large language models can understand and generate human-like text. Machine learning systems can analyze large amounts of data and identify patterns. These technologies are being used in education, healthcare, business, and many other fields.
A bank is a financial institution where people can save and manage their money. Banks also provide loans to individuals and businesses. However, a river bank is the land beside a river. The word "bank" therefore has different meanings depending on the context.
The batsman moved quickly toward the bank after hitting the ball. The ball landed near the river bank, and the players had to retrieve it. Later that day, the player visited a bank to deposit his match earnings.
"""


chunks = text_splitter.create_documents(text)


print("Number of chunks:", len(chunks))
print(chunks)
