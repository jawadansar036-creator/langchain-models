from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


model=ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

class Person(BaseModel):
    name: str = Field(description="The person's full name")
    age: int = Field(description="The person's age")
    city: str = Field(description="Name of the city person belongs to ")
    
parser = PydanticOutputParser(
    pydantic_object=Person
)

template = PromptTemplate(
    template='Generate the name, age and city of a fictional {place} person\n {format_instructions}',
    input_variables=["place"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)

chain = template | model | parser 
result = chain.invoke({'place':'pakistan'})

# prompt = template.invoke({'place':'pakistan'})
# result = model.invoke(prompt)
# final_result = parser.parse(result.content)

print(result)