from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)

prompt = PromptTemplate(
    template='Answer the following question {question} from the following text {text}',
    input_variables=['question','text']
)

parser = StrOutputParser()

url = 'https://www.amazon.com/Ninja-Kitchen-BL770-Processor-Smoothies/dp/B00939I7EK/ref=sr_1_4?_encoding=UTF8&content-id=amzn1.sym.20c14023-dc4e-49db-9d00-9553df5d755b&dib=eyJ2IjoiMSJ9.AwEdYuD2ti2DiFVtVRrNB2jLqyKdcLM5Ajwn-cgS2TAhq1E6rx9eBJhQdO6lyUST2fK82g3zuPcKII55iaT5XhwMeHmmoMExMruwUQRKtNvKKulKnMAzzOx1bpQx8vDB_AIopzMHUQHHJsngyuYjMoQQPtESttXG3JwX54KO_JQfGxi5BcfRAzBI5J-GTAF0qin99WFrURxUZYRhbVIpy3Q2oXnOcWnKzGD0VuBZPoBvizmg_NBEZeu1t3NN9vPGDEfJAWM1qf-OM6l2Ugur7MNlamXAYmE9qtqcHcxOjRU.63m6VCSgNM0xUBmYjS7Y1VxWuek92IuratbnBosLyDc&dib_tag=se&keywords=kitchen&pd_rd_r=8c8d2562-572d-4cfd-bec0-4fb1d50ebb8a&pd_rd_w=5I6HE&pd_rd_wg=Tw3i6&qid=1787914523&sr=8-4&th=1'

loader = WebBaseLoader(url)

docs = loader.load()

chain = prompt | model | parser

result = chain.invoke({'question':'What is the price of the product?','text':docs[0].page_content})

print(result)

