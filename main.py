import os
from pathlib import Path

import streamlit as st

from langchain_huggingface import (
    HuggingFaceEmbeddings,
    HuggingFaceEndpoint,
    ChatHuggingFace
)

from langchain_community.vectorstores import FAISS

from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough
)

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ============================================================
# STREAMLIT PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="VM0042 Agent",
    page_icon="🌱",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🌱 VM0042 Question Answering System")

st.write(
    "Ask questions about the VM0042 Improved Agricultural "
    "Land Management methodology."
)


# ============================================================
# HUGGING FACE TOKEN
# ============================================================

# Streamlit Cloud:
# Go to:
# Settings → Secrets
#
# Add:
# HF_TOKEN = "your_huggingface_token"

HF_TOKEN = st.secrets.get("HF_TOKEN", os.getenv("HF_TOKEN"))

if not HF_TOKEN:
    st.error(
        "Hugging Face token is missing. "
        "Add HF_TOKEN to Streamlit Cloud Secrets."
    )
    st.stop()


# ============================================================
# 1. LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embeddings():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return embeddings


embeddings = load_embeddings()


# ============================================================
# 2. LOAD FAISS VECTOR STORE
# ============================================================

@st.cache_resource
def load_vector_store(_embeddings):

    # Path relative to main.py
    vector_store_path = Path(__file__).parent / "vectore_store"

    # Check folder exists
    if not vector_store_path.exists():

        raise FileNotFoundError(
            f"Vector store folder not found: "
            f"{vector_store_path}"
        )

    # Check FAISS files
    index_file = vector_store_path / "index.faiss"
    pickle_file = vector_store_path / "index.pkl"

    if not index_file.exists():

        raise FileNotFoundError(
            f"Missing index.faiss in {vector_store_path}"
        )

    if not pickle_file.exists():

        raise FileNotFoundError(
            f"Missing index.pkl in {vector_store_path}"
        )

    # Load FAISS
    vector_store = FAISS.load_local(
        str(vector_store_path),
        _embeddings,
        allow_dangerous_deserialization=True
    )

    return vector_store


try:

    vector_store = load_vector_store(embeddings)

except Exception as e:

    st.error("Unable to load the FAISS vector store.")

    st.exception(e)

    st.stop()


# ============================================================
# 3. CREATE RETRIEVER
# ============================================================

retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 4
    }
)


# ============================================================
# 4. HUGGING FACE LLM
# ============================================================

@st.cache_resource
def load_llm():

    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        task="text-generation",
        max_new_tokens=512,
        temperature=0.1,
        huggingfacehub_api_token=HF_TOKEN
    )

    chat_model = ChatHuggingFace(
        llm=llm
    )

    return chat_model


chat_model = load_llm()


# ============================================================
# 5. PROMPT
# ============================================================

prompt = PromptTemplate(
    template="""
You are an expert assistant for the VM0042
Improved Agricultural Land Management methodology.

Answer the user's question using ONLY the information
provided in the context.

Do not make up information.

If the answer is not available in the context,
say:

"I could not find this information in the VM0042 documents."

Give a clear and concise answer.

Context:
{context}

Question:
{question}

Answer:
""",
    input_variables=[
        "context",
        "question"
    ]
)


# ============================================================
# 6. FORMAT DOCUMENTS
# ============================================================

def format_docs(retrieved_docs):

    context_text = "\n\n".join(
        doc.page_content
        for doc in retrieved_docs
    )

    return context_text


# ============================================================
# 7. PARALLEL CHAIN
# ============================================================

parallel_chain = RunnableParallel(
    {
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    }
)


# ============================================================
# 8. OUTPUT PARSER
# ============================================================

parser = StrOutputParser()


# ============================================================
# 9. MAIN RAG CHAIN
# ============================================================

main_chain = (
    parallel_chain
    | prompt
    | chat_model
    | parser
)


# ============================================================
# 10. STREAMLIT QUESTION INPUT
# ============================================================

question = st.text_input(
    "🔎 Ask your question:",
    placeholder="Example: What is the applicability of VM0042?"
)


# ============================================================
# 11. RUN RAG
# ============================================================

if question:

    with st.spinner("Searching VM0042 documents..."):

        try:

            answer = main_chain.invoke(question)

            st.subheader("🤖 Answer")

            st.write(answer)

        except Exception as e:

            st.error("Error while generating the answer.")

            st.exception(e)


# ============================================================
# 12. SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🌱 VM0042 Agent")

    st.write(
        "This application uses:"
    )

    st.write(
        """
        - 📚 FAISS Vector Database
        - 🔎 Semantic Search
        - 🤗 Hugging Face Embeddings
        - 🦙 Llama 3.1
        - 🔗 LangChain
        """
    )

    st.divider()

    st.write(
        "Vector store: vectore_store"
    )
