import os
from pathlib import Path

import streamlit as st

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
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

st.title("🌱 VM0042 Question Answering System")


# ============================================================
# OPENROUTER API KEY
# ============================================================

try:
    OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]
except Exception:
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:

    st.error(
        "❌ OPENROUTER_API_KEY not found.\n\n"
        "Add it in Streamlit Cloud → Manage app → Settings → Secrets."
    )

    st.stop()


# ============================================================
# EMBEDDINGS
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
# VECTOR STORE
# ============================================================

VECTOR_STORE_PATH = (
    Path(__file__).parent / "vector_store"
)

FAISS_INDEX = VECTOR_STORE_PATH / "index.faiss"
FAISS_PICKLE = VECTOR_STORE_PATH / "index.pkl"


if not VECTOR_STORE_PATH.exists():

    st.error(
        f"❌ Vector store folder not found:\n\n"
        f"{VECTOR_STORE_PATH}"
    )

    st.stop()


if not FAISS_INDEX.exists():

    st.error("❌ index.faiss not found.")

    st.stop()


if not FAISS_PICKLE.exists():

    st.error("❌ index.pkl not found.")

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
# OPENROUTER LLM
# ============================================================

@st.cache_resource
def load_llm():
    return ChatOpenAI(
        model="meta-llama/llama-3.3-70b-instruct:free",
        temperature=0.1,
        max_tokens=512,
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "https://vm0042-agent-t9agaafqxbefko68wxzrp7.streamlit.app",
            "X-Title": "VM0042 Question Answering System"
        }
    )


try:

    chat_model = load_llm()

except Exception as e:

    st.error("❌ Failed to initialize OpenRouter LLM.")

    st.exception(e)

    st.stop()


# ============================================================
# PROMPT
# ============================================================

prompt = PromptTemplate(
    template="""
You are a technical assistant specialized in the Verra VM0042
Improved Agricultural Land Management methodology.

Answer the user's question using ONLY the provided VM0042
document context.

RULES:

1. Use ONLY the provided context.

2. Do not use outside knowledge.

3. Do not make assumptions.

4. Do not invent VM0042 requirements, definitions,
   equations, values, eligibility criteria, or project activities.

5. If the answer cannot be found in the context, respond exactly:

"I don't know based on the provided VM0042 documents."

6. For eligibility questions:
   - Give only requirements present in the context.
   - Mention the relevant section when available.

7. For project activity questions:
   - List only activities explicitly supported by the context.

8. For equations:
   - Use only equations present in the context.
   - Explain the variables.
   - Show calculations step by step.
   - Do not create equations.

9. If documents contain conflicting information,
   clearly mention the conflict.

10. Keep the answer concise and factual.

11. If source or page information is available,
    mention it.

------------------------------------------------------------
VM0042 DOCUMENT CONTEXT
------------------------------------------------------------

{context}

------------------------------------------------------------
USER QUESTION
------------------------------------------------------------

{question}

------------------------------------------------------------
ANSWER
------------------------------------------------------------
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

Content:
{doc.page_content}
"""
        )

    return "\n\n".join(formatted)


# ============================================================
# RETRIEVAL
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
# QUESTION
# ============================================================

question = st.text_input(
    "🔎 Ask your question about VM0042",
    placeholder="Example: What is VM0042?"
)


# ============================================================
# ANSWER
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

    st.write("LLM: OpenRouter Free Models")
    st.write("Embeddings: all-MiniLM-L6-v2")
    st.write("Vector Store: FAISS")
    st.write("Retriever: MMR")
