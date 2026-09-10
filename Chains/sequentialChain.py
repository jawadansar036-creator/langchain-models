from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

prompt1 = PromptTemplate(
    template='Genarate a detail report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Genarate a 5 point summmary on following text\n {text}',
    input_variables=['text']
)

model=ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({'topic':'Unemployement in pakistan'})

print(result)

chain.get_graph().print_ascii()