import os
from pathlib import Path

import streamlit as st

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="VM0042 AI Agent",
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 VM0042 AI Agent")
st.caption(
    "AI-powered question answering system for Verra VM0042."
)


# ============================================================
# OPENROUTER API KEY
# ============================================================

try:
    OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]
except Exception:
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


if not OPENROUTER_API_KEY:
    st.error(
        "OPENROUTER_API_KEY is missing. "
        "Add it to Streamlit Cloud → Settings → Secrets."
    )
    st.stop()


# ============================================================
# PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

VECTOR_STORE_PATH = BASE_DIR / "vector_store"

INDEX_FILE = VECTOR_STORE_PATH / "index.faiss"
PKL_FILE = VECTOR_STORE_PATH / "index.pkl"


if not INDEX_FILE.exists():
    st.error(f"Missing FAISS file: {INDEX_FILE}")
    st.stop()

if not PKL_FILE.exists():
    st.error(f"Missing FAISS file: {PKL_FILE}")
    st.stop()


# ============================================================
# EMBEDDINGS
# ============================================================

@st.cache_resource
def load_embeddings():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return embeddings


# ============================================================
# VECTOR STORE
# ============================================================

@st.cache_resource
def load_vector_store():

    embeddings = load_embeddings()

    db = FAISS.load_local(
        str(VECTOR_STORE_PATH),
        embeddings,
        allow_dangerous_deserialization=True
    )

    return db


vector_store = load_vector_store()


# ============================================================
# RETRIEVER
# ============================================================

@st.cache_resource
def load_retriever():

    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 6
        }
    )


retriever = load_retriever()


# ============================================================
# OPENROUTER LLM
# ============================================================

@st.cache_resource
def load_llm():

    llm = ChatOpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",

        # Use a specific model instead of openrouter/free
        model="meta-llama/Llama-3.3-70B-Instruct-evals",

        temperature=0,

        max_tokens=1000,

        default_headers={
            "HTTP-Referer":
                "https://vm0042-agent-t9agaafqxbefko68wxzrp7.streamlit.app",

            "X-Title":
                "VM0042 AI Agent"
        }
    )

    return llm


llm = load_llm()


# ============================================================
# PROMPT
# ============================================================

prompt = ChatPromptTemplate.from_template(
    """
You are a specialized AI assistant for the Verra VM0042
Improved Agricultural Land Management methodology.

You are answering questions using the retrieved VM0042
document content.

STRICT RULES:

1. Answer the user's question directly.
2. Use the provided VM0042 context as your factual source.
3. Do not invent information.
4. Do not add requirements that are not present in the context.
5. Do not use unrelated general knowledge.
6. If the answer is clearly available in the context,
   explain it naturally.
7. For lists, use bullet points.
8. For eligibility questions, list the relevant eligibility
   requirements from the documents.
9. For project activity questions, list the activities supported
   by the documents.
10. For definitions, give a clear definition based on the documents.
11. If the retrieved context does not contain enough information,
    respond exactly:

I don't know based on the provided VM0042 documents.

12. Do not simply copy large sections of the documents.
13. Summarize and explain the information.
14. Mention the source document/page when available.

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
)


# ============================================================
# FORMAT DOCUMENTS
# ============================================================

def format_documents(documents):

    context_parts = []

    for i, doc in enumerate(documents, 1):

        metadata = doc.metadata or {}

        source = (
            metadata.get("source")
            or metadata.get("file_name")
            or metadata.get("filename")
            or "VM0042 document"
        )

        page = metadata.get("page")

        if page is not None:
            source_name = f"{source}, page {page}"
        else:
            source_name = source

        text = doc.page_content

        context_parts.append(
            f"""
SOURCE {i}
Document: {source_name}

{text}
"""
        )

    return "\n".join(context_parts)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🌱 VM0042 AI Agent")

    st.write("📚 Knowledge: VM0042 PDFs")
    st.write("🔎 Retrieval: FAISS")
    st.write("🧠 LLM: OpenRouter")
    st.write("🔑 Authentication: OpenRouter API")

    st.divider()

    if st.button("🗑️ Clear Chat"):

        st.session_state.messages = []

        st.rerun()


# ============================================================
# SHOW CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask anything about VM0042..."
)


if question:

    # ========================================================
    # USER MESSAGE
    # ========================================================

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):

        st.markdown(question)


    # ========================================================
    # ASSISTANT
    # ========================================================

    with st.chat_message("assistant"):

        try:

            with st.spinner("🔎 Searching VM0042 documents..."):

                documents = retriever.invoke(question)


            if not documents:

                answer = (
                    "I don't know based on the provided "
                    "VM0042 documents."
                )

            else:

                context = format_documents(documents)

                with st.spinner("🧠 Generating answer..."):

                    formatted_prompt = prompt.format_messages(
                        context=context,
                        question=question
                    )

                    response = llm.invoke(
                        formatted_prompt
                    )

                    answer = response.content

                    if not answer:

                        answer = (
                            "I don't know based on the provided "
                            "VM0042 documents."
                        )


            # =================================================
            # DISPLAY ANSWER
            # =================================================

            st.markdown(answer)


            # =================================================
            # SOURCES
            # =================================================

            with st.expander("📚 Retrieved VM0042 Sources"):

                for i, doc in enumerate(documents, 1):

                    metadata = doc.metadata or {}

                    source = (
                        metadata.get("source")
                        or metadata.get("file_name")
                        or metadata.get("filename")
                        or "VM0042 document"
                    )

                    page = metadata.get("page")

                    if page is not None:

                        st.markdown(
                            f"**{i}. {source} — Page {page}**"
                        )

                    else:

                        st.markdown(
                            f"**{i}. {source}**"
                        )

                    preview = doc.page_content

                    if len(preview) > 600:

                        preview = preview[:600] + "..."

                    st.caption(preview)


        except Exception as e:

            answer = f"""
**Error while generating the answer**

`{str(e)}`
"""

            st.error(answer)


    # ========================================================
    # SAVE ASSISTANT MESSAGE
    # ========================================================

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )
