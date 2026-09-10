from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv

load_dotenv()

prompt1 = PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=["topic"]
)

model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)

parser = StrOutputParser()

prompt2 = PromptTemplate(
    template="Explain the following joke: {text}",
    input_variables=["text"]
)

chain = RunnableSequence(prompt1,model,parser,prompt2,model,parser)

result = chain.invoke({"topic": "India"})

print(result)