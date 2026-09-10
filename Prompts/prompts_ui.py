from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
from langchain_core.prompts import PromptTemplate , load_prompt
from dotenv import load_dotenv

load_dotenv()

# Gemini model
model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=1.5
)

# App title
st.title("Research Tool")

# Dropdown 1: Research Paper
paper_input = st.selectbox(
    "Select Research Paper Name",
    [
        "Attention Is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "GPT-3: Language Models are Few-Shot Learners",
        "Diffusion Models Beat GANs on Image Synthesis"
    ]
)

# Dropdown 2: Explanation Style
style_input = st.selectbox(
    "Select Explanation Style",
    [
        "Beginner-Friendly",
        "Technical",
        "Code-Oriented",
        "Mathematical"
    ]
)

# Dropdown 3: Explanation Length
length_input = st.selectbox(
    "Select Explanation Length",
    [
        "Short (1-2 paragraphs)",
        "Medium (3-5 paragraphs)",
        "Long (detailed explanation)"
    ]
)

template = load_prompt('template.json')

if st.button('Summarize'):
    chain = template | model
    result = chain.invoke({
    "paper_input": paper_input,
    "style_input": style_input,
    "length_input": length_input
})
    st.write(result.content[0]['text'])
