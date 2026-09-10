from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda

from pydantic import BaseModel, Field
from typing import Literal

from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)

parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(
        description="Give the sentiment of the feedback"
    )

parser1 = PydanticOutputParser(
    pydantic_object=Feedback
)

prompt1 = PromptTemplate(
    template="""
Classify the sentiment of feedback text into positive or negative.

Feedback:
{feedback}

{format_instructions}
""",
    input_variables=["feedback"],
    partial_variables={
        "format_instructions": parser1.get_format_instructions()
    }
)

classifier_chain = prompt1 | model | parser1

prompt2 = PromptTemplate(
    template="""Write an appropriate response to the positive feedback\n{feedback}""",
    input_variables=["feedback"]
)

prompt3 = PromptTemplate(
    template="""Write an appropriate response to the negative feedback\n{feedback}""",
    input_variables=["feedback"]
)

classifier_chain = RunnableParallel(
    {
        "feedback": lambda x: x["feedback"],
        "sentiment": classifier_chain
    }
)

branch_chain = RunnableBranch(
    (lambda x: x["sentiment"].sentiment == "positive",prompt2 | model | parser ),
    ( lambda x: x["sentiment"].sentiment == "negative", prompt3 | model | parser ),
    RunnableLambda(lambda x: "Could not find sentiment")
)

chain = classifier_chain | branch_chain

result = chain.invoke(
    {
        "feedback": "I really love this application. It is very useful and easy to use."
    }
)

print(result)

chain.get_graph().print_ascii()