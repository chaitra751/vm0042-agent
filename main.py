import os
from pathlib import Path

import streamlit as st

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="VM0042 AI Agent",
    page_icon="🌱",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🌱 VM0042 AI Agent")
st.caption(
    "Ask questions about the Verra VM0042 Improved Agricultural Land Management methodology."
)


# ============================================================
# OPENROUTER API KEY
# ============================================================

OPENROUTER_API_KEY = (
    st.secrets.get("OPENROUTER_API_KEY")
    if "OPENROUTER_API_KEY" in st.secrets
    else os.getenv("OPENROUTER_API_KEY")
)

if not OPENROUTER_API_KEY:
    st.error("OPENROUTER_API_KEY is not configured in Streamlit Secrets.")
    st.stop()


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).parent
VECTOR_STORE_PATH = BASE_DIR / "vector_store"


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

@st.cache_resource
def load_embeddings():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# ============================================================
# LOAD FAISS
# ============================================================

@st.cache_resource
def load_vector_store():

    index_file = VECTOR_STORE_PATH / "index.faiss"
    pickle_file = VECTOR_STORE_PATH / "index.pkl"

    if not index_file.exists():
        st.error(f"FAISS index not found: {index_file}")
        st.stop()

    if not pickle_file.exists():
        st.error(f"FAISS metadata not found: {pickle_file}")
        st.stop()

    embeddings = load_embeddings()

    vector_store = FAISS.load_local(
        str(VECTOR_STORE_PATH),
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vector_store


# ============================================================
# LOAD RETRIEVER
# ============================================================

@st.cache_resource
def load_retriever():

    vector_store = load_vector_store()

    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 6,
            "fetch_k": 30,
            "lambda_mult": 0.7
        }
    )


retriever = load_retriever()


# ============================================================
# OPENROUTER LLM
# ============================================================

@st.cache_resource
def load_llm():

    return ChatOpenAI(
        model="openrouter/free",
        temperature=0,
        max_tokens=800,
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "https://vm0042-agent-t9agaafqxbefko68wxzrp7.streamlit.app",
            "X-Title": "VM0042 AI Agent"
        }
    )


llm = load_llm()


# ============================================================
# PROMPT
# ============================================================

prompt = ChatPromptTemplate.from_template(
    """
You are an AI assistant specialized in the Verra VM0042
Improved Agricultural Land Management methodology.

Your task is to answer the user's question using ONLY the
provided VM0042 document context.

IMPORTANT RULES:

1. Use only information contained in the provided context.
2. Do not use your general knowledge.
3. Do not invent information.
4. Do not assume information that is not explicitly present.
5. Give a direct and natural chatbot-style answer.
6. If the question asks for a list, provide a clear bullet list.
7. If the question asks about eligibility criteria, provide only
   eligibility requirements supported by the context.
8. If the question asks for project activities, list only activities
   explicitly supported by the context.
9. If the question asks about an equation, explain the equation,
   variables and calculation only when present in the context.
10. If the context does not contain enough information, say:

"I don't know based on the provided VM0042 documents."

11. Do not mention that you are an AI unless necessary.
12. Keep the answer concise but complete.
13. At the end, provide the source document/page when metadata is available.

DOCUMENT CONTEXT:
-----------------
{context}
-----------------

USER QUESTION:
{question}

ANSWER:
"""
)


# ============================================================
# FORMAT DOCUMENTS
# ============================================================

def format_documents(documents):

    formatted = []

    for i, doc in enumerate(documents, start=1):

        metadata = doc.metadata or {}

        source = (
            metadata.get("source")
            or metadata.get("file_name")
            or metadata.get("filename")
            or "VM0042 document"
        )

        page = metadata.get("page")

        if page is not None:
            source_info = f"{source}, page {page}"
        else:
            source_info = source

        formatted.append(
            f"""
SOURCE {i}
Document: {source_info}

Content:
{doc.page_content}
"""
        )

    return "\n".join(formatted)


# ============================================================
# CHAT HISTORY
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🌱 VM0042")

    st.write("**AI Agent Configuration**")

    st.write("📚 Knowledge Base: VM0042 PDFs")
    st.write("🔎 Retrieval: FAISS")
    st.write("🧠 LLM: OpenRouter")
    st.write("🔑 API Key: Configured")

    st.divider()

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()


# ============================================================
# DISPLAY PREVIOUS CHAT
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ============================================================
# USER QUESTION
# ============================================================

question = st.chat_input(
    "Ask a question about VM0042..."
)


if question:

    # --------------------------------------------------------
    # USER MESSAGE
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)


    # --------------------------------------------------------
    # ASSISTANT
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner("Searching VM0042 documents..."):

            try:

                # Retrieve relevant documents
                documents = retriever.invoke(question)

                if not documents:

                    answer = (
                        "I don't know based on the provided "
                        "VM0042 documents."
                    )

                else:

                    context = format_documents(documents)

                    # ------------------------------------------------
                    # SEND CONTEXT + QUESTION TO OPENROUTER
                    # ------------------------------------------------

                    messages = prompt.format_messages(
                        context=context,
                        question=question
                    )

                    response = llm.invoke(messages)

                    answer = response.content

                    if not answer:
                        answer = (
                            "I don't know based on the provided "
                            "VM0042 documents."
                        )

                st.markdown(answer)

                # ----------------------------------------------------
                # SOURCE DOCUMENTS
                # ----------------------------------------------------

                with st.expander("📚 Retrieved Sources"):

                    for i, doc in enumerate(documents, start=1):

                        metadata = doc.metadata or {}

                        source = (
                            metadata.get("source")
                            or metadata.get("file_name")
                            or metadata.get("filename")
                            or "VM0042 document"
                        )

                        page = metadata.get("page")

                        if page is not None:
                            st.write(
                                f"**{i}. {source} — Page {page}**"
                            )
                        else:
                            st.write(f"**{i}. {source}**")

                        st.caption(
                            doc.page_content[:700] + "..."
                            if len(doc.page_content) > 700
                            else doc.page_content
                        )

            except Exception as e:

                answer = f"Error while generating the answer:\n\n`{e}`"

                st.error(answer)


    # --------------------------------------------------------
    # SAVE ASSISTANT RESPONSE
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )
