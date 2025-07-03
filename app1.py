import os
from dotenv import load_dotenv
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

# Set page config for better layout
st.set_page_config(page_title="🧠 Ask Gemma | LangChain", layout="centered")

# Header section with styling
st.markdown("""
    <h1 style='text-align: center; color: #4A90E2;'>🤖 Ask Anything with Gemma</h1>
    <p style='text-align: center; font-size:18px;'>Built with LangChain, Ollama & Streamlit</p>
    <hr style='margin-bottom: 30px;'>
""", unsafe_allow_html=True)

# Set up LLM and prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Provide concise answers to the user's questions."),
    ("user", "Question:{question}")
])
llm = Ollama(model="gemma:2b")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

# Input section
st.markdown("### 💬 What do you want to ask?")
new_question = st.text_input("", placeholder="Type your question here...")

# Button and output logic
if st.button("🚀 Get Answer"):
    if new_question.strip():
        with st.spinner("Thinking... 🤔"):
            answer = chain.invoke({"question": new_question})

        with st.container():
            st.markdown(f"""
            <div style='background-color: #f0f8ff; padding: 15px; border-radius: 10px; margin-top: 20px;'>
                <h4 style='color: #333;'>🧠 <b>Question:</b></h4>
                <p style='font-size: 16px;'>{new_question}</p>
                <h4 style='color: #333; margin-top: 15px;'>📘 <b>Answer:</b></h4>
                <p style='font-size: 16px; color: #007BFF;'>{answer}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("Please enter a question before clicking 'Get Answer'.")

# Footer
st.markdown("""
<hr>
<p style='text-align: center; font-size: 13px; color: gray;'>
Built with ❤️ by Rohit | Powered by LangChain, Ollama & Streamlit
</p>
""", unsafe_allow_html=True)
