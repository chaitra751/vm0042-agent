import streamlit as st
from pathlib import Path
import base64
import streamlit.components.v1 as components
import faiss


import streamlit as st

st.title("VM0042 Agent")

st.write("Hi! Welcome to the VM0042 Question Answering System.")

st.write(index)

# Optional: show basic information
st.write("Number of vectors:", index.ntotal)
st.write("Vector dimension:", index.d)
