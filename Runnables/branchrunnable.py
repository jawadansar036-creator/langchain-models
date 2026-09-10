from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnablePassthrough,RunnableLambda,RunnableBranch
from dotenv import load_dotenv

load_dotenv()

def word_count(text):
    return len(text.split())

model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Summarize the following text {text}",
    input_variables=["text"]
)

report_chain = RunnableSequence(prompt1,model,parser)

branch_chain = RunnableBranch(
    (lambda x: len(x.split())>500,RunnableSequence(prompt2,model,parser)),
    RunnablePassthrough()
)

final_chain = RunnableSequence(report_chain,branch_chain)

result = final_chain.invoke({'topic':'Russia vs Ukraine'})

print(result)

