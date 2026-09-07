from pathlib import Path
import base64
import faiss
import streamlit as st
import streamlit.components.v1 as components
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

st.title("VM0042 Agent")
st.write("Hi! Welcome to the VM0042 Question Answering System.")

BASE_DIR = Path(__file__).resolve().parent
VECTOR_STORE_DIR = BASE_DIR / "vector_store"
INDEX_PATH = VECTOR_STORE_DIR / "index.faiss"

if not INDEX_PATH.exists():
    st.error(f"FAISS index not found: {INDEX_PATH}")
    st.stop()

# 1. Initialize the embedding model (Dimension 384)
# If you used OpenAI embeddings originally, use OpenAIEmbeddings() instead
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 2. Load the LangChain FAISS vector store cleanly
try:
    vector_store = FAISS.load_local(
        folder_path=str(VECTOR_STORE_DIR),
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )

    # 3. Display metadata using the underlying FAISS index object
    st.write("Number of vectors:", vector_store.index.ntotal)
    st.write("Vector dimension:", vector_store.index.d)

    # 4. Initialize retriever
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 4}
    )

    retriever_result=retriever.invoke('What is vm0042')
    st.write(retriever_result)
    st.success("Vector store loaded successfully!")

except Exception as e:
    st.error(f"Failed to load vector store: {e}")
