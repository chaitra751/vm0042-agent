import os
from pathlib import Path

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint

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
    "AI assistant for the Verra VM0042 Improved Agricultural Land Management methodology"
)


# ============================================================
# HUGGING FACE TOKEN
# ============================================================

HF_TOKEN = st.secrets.get("HF_TOKEN")

if not HF_TOKEN:
    st.error("❌ HF_TOKEN is missing from Streamlit Secrets.")
    st.info(
        "Go to Streamlit Cloud → App → Settings → Secrets "
        "and add HF_TOKEN."
    )
    st.stop()

HF_TOKEN = HF_TOKEN.strip()


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
# VECTOR STORE PATH & CHECKS
# ============================================================

VECTOR_STORE_PATH = Path(__file__).parent / "vector_store"
FAISS_INDEX = VECTOR_STORE_PATH / "index.faiss"
FAISS_PICKLE = VECTOR_STORE_PATH / "index.pkl"

if not VECTOR_STORE_PATH.exists() or not FAISS_INDEX.exists() or not FAISS_PICKLE.exists():
    st.error(
        f"""
❌ Vector store files missing from:
`{VECTOR_STORE_PATH}`

Make sure both `index.faiss` and `index.pkl` exist in the `vector_store` directory.
"""
    )
    st.stop()


# ============================================================
# LOAD FAISS VECTOR STORE
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
# CREATE RETRIEVER
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
# LOAD LLM VIA HUGGINGFACE ENDPOINT
# ============================================================

MODEL_NAME = "openai/gpt-oss-120b"


@st.cache_resource
def load_llm():
    return HuggingFaceEndpoint(
        repo_id=MODEL_NAME,
        huggingfacehub_api_token=HF_TOKEN,
        temperature=0.1,
        max_new_tokens=512,
    )


try:
    llm = load_llm()
except Exception as e:
    st.error("❌ Failed to initialize Hugging Face LLM endpoint.")
    st.exception(e)
    st.stop()


# ============================================================
# PROMPT
# ============================================================

PROMPT_TEMPLATE = """
You are a technical AI assistant specialized in the Verra VM0042
Improved Agricultural Land Management methodology.

Your task is to answer the user's question using ONLY the information
provided in the VM0042 document context.

IMPORTANT RULES:
1. Use only the provided context.
2. Do not use your own knowledge or outside information.
3. Do not make assumptions.
4. If the answer cannot be found in the context, respond exactly:
"I don't know based on the provided VM0042 documents."
5. Do not invent, modify, or assume any VM0042 requirements, values, or equations.
6. Keep the answer concise and factual.

--------------------------------------------------
VM0042 DOCUMENT CONTEXT
--------------------------------------------------

{context}

--------------------------------------------------
USER QUESTION
--------------------------------------------------

{question}

--------------------------------------------------
ANSWER
--------------------------------------------------
"""


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def retrieve_documents(question):
    try:
        return retriever.invoke(question)
    except Exception as e:
        st.error("❌ Error retrieving documents from FAISS.")
        st.exception(e)
        return []


def format_context(documents):
    if not documents:
        return "No relevant VM0042 documents were found."

    context_parts = []
    for i, doc in enumerate(documents, start=1):
        metadata = doc.metadata or {}
        source = metadata.get("source", "VM0042 document")
        page = metadata.get("page", metadata.get("page_number", ""))
        source_info = f"{source}, page {page}" if page != "" else str(source)

        context_parts.append(
            f"--- DOCUMENT {i} ---\nSource: {source_info}\n\n{doc.page_content}"
        )

    return "\n\n".join(context_parts)


def generate_answer(question, context):
    prompt = PROMPT_TEMPLATE.format(
        context=context,
        question=question
    )
    return llm.invoke(prompt).strip()


# ============================================================
# CHAT HISTORY
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("🌱 VM0042 Agent")
    st.write(
        "Ask questions about the Verra VM0042 "
        "Improved Agricultural Land Management methodology."
    )
    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption(f"LLM: {MODEL_NAME}")
    st.caption("Embeddings: all-MiniLM-L6-v2")
    st.caption("Retriever: FAISS + MMR")


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ============================================================
# USER INPUT & PROCESS QUESTION
# ============================================================

question = st.chat_input("Ask a question about VM0042...")

if question:
    # 1. Add User Message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # 2. Process Assistant Message
    with st.chat_message("assistant"):
        with st.spinner("🔎 Searching VM0042 documents..."):
            documents = retrieve_documents(question)

        if not documents:
            answer = "I don't know based on the provided VM0042 documents."
            st.markdown(answer)
        else:
            context = format_context(documents)

            with st.spinner("🤖 Generating answer..."):
                try:
                    answer = generate_answer(question, context)
                except Exception as e:
                    st.error("❌ Error while generating the answer.")
                    st.exception(e)
                    answer = None

            if answer:
                st.markdown(answer)

                # Show Sources
                with st.expander("📚 Retrieved VM0042 Sources"):
                    for i, doc in enumerate(documents, start=1):
                        metadata = doc.metadata or {}
                        source = metadata.get("source", "VM0042 document")
                        page = metadata.get("page", metadata.get("page_number", ""))
                        page_str = f" — Page {page}" if page != "" else ""
                        st.markdown(f"**Document {i}:** {source}{page_str}")

        # Save Response
        if answer:
            st.session_state.messages.append({"role": "assistant", "content": answer})
