import streamlit as st
from pathlib import Path
import base64
import streamlit.components.v1 as components
import faiss
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


import streamlit as st

st.title("VM0042 Agent")

st.write("Hi! Welcome to the VM0042 Question Answering System.")

BASE_DIR = Path(__file__).resolve().parent
INDEX_PATH = BASE_DIR / "vector_store" / "index.faiss"

if not INDEX_PATH.exists():
    st.error(f"FAISS index not found: {INDEX_PATH}")
    st.stop()

vector_store = faiss.read_index(str(INDEX_PATH))


st.write(vector_store )

# Optional: show basic information
st.write("Number of vectors:", vector_store .ntotal)
st.write("Vector dimension:", vector_store .d)

embeddings = OpenAIEmbeddings()

vector_store = FAISS.load_local(
    "vector_store",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)
