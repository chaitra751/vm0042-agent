import streamlit as st
from pathlib import Path
import base64
import streamlit.components.v1 as components
import faiss


import streamlit as st

st.title("VM0042 Agent")

st.write("Hi! Welcome to the VM0042 Question Answering System.")

# 1. Load the FAISS index
index = faiss.read_index("index.faiss")

# 2. View key index properties
print(f"Total vectors indexed: {index.ntotal}")
print(f"Vector dimension: {index.d}")
print(f"Is index trained? {index.is_trained}")

# 3. Reconstruct specific vectors (if the index type supports reconstruction)
try:
    # Retrieve the first vector in the index (index 0)
    first_vector = index.reconstruct(0)
    print("Sample vector array:", first_vector[:5])  # Prints first 5 dimensions
except Exception as e:
    print("This index type does not support direct vector reconstruction:", e)
