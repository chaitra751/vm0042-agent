import os
from pathlib import Path

import streamlit as st

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

from langchain_community.vectorstores import FAISS

from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough
)

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ============================================================
# PAGE CONFIG
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

st.caption(
    "AI-powered VM0042 document question answering"
)


# ============================================================
# VECTOR STORE PATH
# ============================================================

VECTOR_STORE_PATH = Path(__file__).parent / "vector_store"

FAISS_INDEX = VECTOR_STORE_PATH / "index.faiss"
FAISS_PICKLE = VECTOR_STORE_PATH / "index.pkl"


# ============================================================
# CHECK VECTOR STORE
# ============================================================

if not VECTOR_STORE_PATH.exists():

    st.error(
        f"""
        ❌ Vector store folder not found.

        Expected:

        `{VECTOR_STORE_PATH}`

        Required structure:

        vm0042-agent/
        ├── main.py
        ├── requirements.txt
        └── vector_store/
            ├── index.faiss
            └── index.pkl
        """
    )

    st.stop()


if not FAISS_INDEX.exists():

    st.error(
        f"❌ index.faiss not found: `{FAISS_INDEX}`"
    )

    st.stop()


if not FAISS_PICKLE.exists():

    st.error(
        f"❌ index.pkl not found: `{FAISS_PICKLE}`"
    )

    st.stop()


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

@st.cache_resource
def load_embeddings():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


try:

    embeddings = load_embeddings()

except Exception as e:

    st.error("❌ Failed to load embedding model.")

    st.exception(e)

    st.stop()


# ============================================================
# LOAD FAISS
# ============================================================

@st.cache_resource
def load_vector_store():

    return FAISS.load_local(
        str(VECTOR_STORE_PATH),
        embeddings,
        allow_dangerous_deserialization=True
    )


try:

    vector_store = load_vector_store()

except Exception as e:

    st.error("❌ Unable to load FAISS vector store.")

    st.write(
        "Make sure the FAISS index was created using "
        "sentence-transformers/all-MiniLM-L6-v2."
    )

    st.exception(e)

    st.stop()


# ============================================================
# RETRIEVER
# ============================================================

retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 6,
        "fetch_k": 30,
        "lambda_mult": 0.6
    }
)


# ============================================================
# OLLAMA LLM
# ============================================================

@st.cache_resource
def load_llm():

    return ChatOllama(
        model="llama3.1:8b",
        temperature=0.1,
        num_predict=512
    )


try:

    chat_model = load_llm()

except Exception as e:

    st.error(
        """
        ❌ Could not connect to Ollama.

        Make sure Ollama is running and the model is installed.

        Run:

        ollama pull llama3.1:8b

        Then:

        ollama serve
        """
    )

    st.exception(e)

    st.stop()


# ============================================================
# PROMPT
# ============================================================

prompt = PromptTemplate(
    template="""
You are a technical AI assistant specialized in the Verra
VM0042 Improved Agricultural Land Management methodology.

Answer the user's question using ONLY the VM0042 document
context provided below.

RULES:

1. Use only the provided context.

2. Do not use outside knowledge.

3. Do not make assumptions.

4. Do not invent requirements, values, equations,
   definitions, eligibility criteria, or project activities.

5. If the answer is not available in the context, respond:

"I don't know based on the provided VM0042 documents."

6. For eligibility questions:
   - Give only requirements found in the context.
   - Mention the relevant section when available.

7. For project activity questions:
   - List only activities explicitly supported by the context.

8. For equations:
   - Use only equations present in the context.
   - Explain variables.
   - Show calculations step by step.
   - Do not create equations.

9. If information conflicts between documents,
   mention the conflict.

10. Keep the answer concise and factual.

11. If document name or page information is available,
    mention it.

------------------------------------------------------------

CONTEXT:

{context}

------------------------------------------------------------

QUESTION:

{question}

------------------------------------------------------------

ANSWER:
""",
    input_variables=["context", "question"]
)


# ============================================================
# FORMAT DOCUMENTS
# ============================================================

def format_docs(docs):

    if not docs:

        return "No relevant VM0042 documents were found."

    formatted = []

    for i, doc in enumerate(docs, start=1):

        source = doc.metadata.get(
            "source",
            "Unknown document"
        )

        page = doc.metadata.get(
            "page",
            "Unknown page"
        )

        formatted.append(
            f"""
DOCUMENT {i}
Source: {source}
Page: {page}

{doc.page_content}
"""
        )

    return "\n\n".join(formatted)


# ============================================================
# RETRIEVAL CHAIN
# ============================================================

parallel_chain = RunnableParallel(
    {
        "context": (
            retriever
            | RunnableLambda(format_docs)
        ),
        "question": RunnablePassthrough()
    }
)


# ============================================================
# MAIN RAG CHAIN
# ============================================================

main_chain = (
    parallel_chain
    | prompt
    | chat_model
    | StrOutputParser()
)


# ============================================================
# USER INPUT
# ============================================================

question = st.text_input(
    "🔎 Ask your question about VM0042",
    placeholder="Example: What is VM0042?"
)


# ============================================================
# GENERATE ANSWER
# ============================================================

if question:

    with st.spinner(
        "🔎 Searching VM0042 documents..."
    ):

        try:

            answer = main_chain.invoke(question)

            st.subheader("🤖 Answer")

            st.write(answer)

        except Exception as e:

            st.error(
                "❌ Error while generating the answer."
            )

            st.exception(e)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🌱 VM0042 Agent")

    st.write("LLM: Llama 3.1 8B")
    st.write("LLM Provider: Ollama")
    st.write("Embeddings: all-MiniLM-L6-v2")
    st.write("Vector Store: FAISS")
    st.write("Retriever: MMR")
