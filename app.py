import streamlit as st
from src.loader import load_and_split_documents
from src.vectorstore import create_vectorstore, load_vectorstore
from src.rag_chain import build_rag_chain
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import streamlit as st

load_dotenv()

# Load or create vectorstore
persist_dir = "embeddings/"
try:
    vectorstore = load_vectorstore(persist_dir)
except:
    st.warning("Creating vectorstore. This may take a few minutes...")
    docs = load_and_split_documents()
    vectorstore = create_vectorstore(docs, persist_dir=persist_dir)

# Initialize LLM
groq_api_key = os.getenv("GROQ_API_KEY") or st.secrets["GROQ_API_KEY"]

llm = ChatGroq(groq_api_key=groq_api_key, model="llama-3.1-8b-instant", temperature=0.2, max_tokens=1000)

# Build RAG chain
rag_chain = build_rag_chain(vectorstore, llm)

st.title("AI Doctor at Home")

user_question = st.text_input("Ask a medical question:")

if user_question:
    answer = rag_chain.invoke(user_question)
    st.subheader("Answer:")
    st.write(answer)