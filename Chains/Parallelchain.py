
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

from dotenv import load_dotenv

load_dotenv()


# Models
model1 = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)

model2 = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)


# Prompt 1: Generate notes
prompt1 = PromptTemplate(
    template="""Generate short and simple notes from the following text: {text}""",
    input_variables=["text"]
)


# Prompt 2: Generate quiz questions
prompt2 = PromptTemplate(
    template="""Generate 5 short questions from the following text: {text}""",
    input_variables=["text"]
)


# Prompt 3: Merge notes and quiz
prompt3 = PromptTemplate(
    template="""Merge the provided notes and quiz into a single document.

Notes:{notes}

Quiz:{quiz}""",
input_variables=["notes", "quiz"]
)

parser = StrOutputParser()



parallel_chain = RunnableParallel(
    {
        "notes": prompt1 | model1 | parser,
        "quiz": prompt2 | model2 | parser
    }
)


merge_chain = prompt3 | model1 | parser


chain = parallel_chain | merge_chain

text = """
Support vector machines (SVMs) are a set of supervised learning methods
used for classification, regression and outliers detection.

The advantages of support vector machines are:

Effective in high dimensional spaces.

Still effective in cases where number of dimensions is greater than the
number of samples.

Uses a subset of training points in the decision function (called support
vectors), so it is also memory efficient.

Versatile: different Kernel functions can be specified for the decision
function. Common kernels are provided, but it is also possible to specify
custom kernels.

The disadvantages of support vector machines include:

If the number of features is much greater than the number of samples,
avoid over-fitting in choosing Kernel functions and regularization term
is crucial.

SVMs do not directly provide probability estimates, these are calculated
using an expensive five-fold cross-validation.

The support vector machines in scikit-learn support both dense
(numpy.ndarray and convertible to that by numpy.asarray) and sparse
(any scipy.sparse) sample vectors as input.
"""

result = chain.invoke({"text": text})

print(result)

chain.get_graph().print_ascii()