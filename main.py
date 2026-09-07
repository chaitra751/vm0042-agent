import streamlit as st
from pathlib import Path
import base64
import streamlit.components.v1 as components
import faiss
from langchain_community.vectorstores import FAISS


import streamlit as st

st.title("VM0042 Agent")

st.write("Hi! Welcome to the VM0042 Question Answering System.")

# Load the FAISS index
vector_store = faiss.read_index("index.faiss")

st.write(vector_store )

# Optional: show basic information
st.write("Number of vectors:", vector_store .ntotal)
st.write("Vector dimension:", vector_store .d)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
