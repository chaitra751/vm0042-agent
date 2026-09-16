import os
from pathlib import Path

import streamlit as st

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from huggingface_hub import InferenceClient


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
# VECTOR STORE PATH
# ============================================================

VECTOR_STORE_PATH = Path(__file__).parent / "vector_store"


# ============================================================
# CHECK VECTOR STORE
# ============================================================

if not VECTOR_STORE_PATH.exists():

    st.error(
        f"""
❌ Vector store folder not found.

Expected location:

`{VECTOR_STORE_PATH}`

Your GitHub repository should contain:

vm0042-agent/
├── main.py
├── requirements.txt
└── vector_store/
    ├── index.faiss
    └── index.pkl
"""
    )

    st.stop()


# ============================================================
# CHECK FAISS FILES
# ============================================================

FAISS_INDEX = VECTOR_STORE_PATH / "index.faiss"
FAISS_PICKLE = VECTOR_STORE_PATH / "index.pkl"


if not FAISS_INDEX.exists():

    st.error(
        f"❌ `index.faiss` not found inside:\n\n"
        f"`{VECTOR_STORE_PATH}`"
    )

    st.stop()


if not FAISS_PICKLE.exists():

    st.error(
        f"❌ `index.pkl` not found inside:\n\n"
        f"`{VECTOR_STORE_PATH}`"
    )

    st.stop()


# ============================================================
# LOAD FAISS VECTOR STORE
# ============================================================

@st.cache_resource
def load_vector_store():

    vector_store = FAISS.load_local(
        str(VECTOR_STORE_PATH),
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vector_store


try:

    vector_store = load_vector_store()

except Exception as e:

    st.error("❌ Unable to load FAISS vector store.")

    st.write(
        "Make sure `index.faiss` and `index.pkl` were created "
        "using the same embedding model:"
    )

    st.code(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

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
# HUGGING FACE INFERENCE CLIENT
# ============================================================

@st.cache_resource
def load_llm():

    client = InferenceClient(
        token=HF_TOKEN
    )

    return client


try:

    client = load_llm()

except Exception as e:

    st.error("❌ Failed to initialize Hugging Face client.")
    st.exception(e)
    st.stop()


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = "openai/gpt-oss-120b"


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

5. Do not invent, modify, or assume any VM0042:
   - requirements
   - values
   - equations
   - variables
   - definitions
   - eligibility criteria
   - project activities
   - monitoring requirements

6. For equations or calculations:
   - Identify the equation from the context.
   - Write the equation clearly.
   - Explain the variables.
   - Substitute the provided values.
   - Show the calculation.
   - Give the final result with the correct unit.
   - Never create an equation that is not present in the context.

7. If multiple sections are relevant, combine them carefully.

8. If the context contains conflicting information, clearly mention
   the conflict instead of choosing one by assumption.

9. For questions about:
   - project activity eligibility
   - applicability
   - baseline
   - project boundaries
   - additionality
   - leakage
   - emission reductions
   - monitoring
   - project activities

   use only the requirements explicitly available in the context.

10. Answer naturally like a chatbot.

11. Do not simply copy large sections of the document.

12. Keep the answer concise and factual.

13. When the context contains page numbers or section names, mention
    them when useful.

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
# RETRIEVE DOCUMENTS
# ============================================================

def retrieve_documents(question):

    try:

        documents = retriever.invoke(question)

        return documents

    except Exception as e:

        st.error("❌ Error retrieving documents from FAISS.")
        st.exception(e)

        return []


# ============================================================
# FORMAT CONTEXT
# ============================================================

def format_context(documents):

    if not documents:

        return "No relevant VM0042 documents were found."

    context_parts = []

    for i, doc in enumerate(documents, start=1):

        metadata = doc.metadata or {}

        source = metadata.get(
            "source",
            "VM0042 document"
        )

        page = metadata.get(
            "page",
            metadata.get("page_number", "")
        )

        if page != "":
            source_info = f"{source}, page {page}"
        else:
            source_info = str(source)

        context_parts.append(
            f"""
--- DOCUMENT {i} ---
Source: {source_info}

{doc.page_content}
"""
        )

    return "\n\n".join(context_parts)


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(question, context):

    prompt = PROMPT_TEMPLATE.format(
        context=context,
        question=question
    )

    response = client.chat.completions.create(
        model=MODEL_NAME,

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],

        max_tokens=512,

        temperature=0.1
    )

    return response.choices[0].message.content.strip()


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

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()

    


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ============================================================
# USER INPUT
# ============================================================

question = st.chat_input(
    "Ask a question about VM0042..."
)


# ============================================================
# PROCESS QUESTION
# ============================================================

if question:

    # --------------------------------------------------------
    # SHOW USER MESSAGE
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
    # RETRIEVE DOCUMENTS
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "🔎 Searching VM0042 documents..."
        ):

            documents = retrieve_documents(
                question
            )


        # ----------------------------------------------------
        # CHECK RETRIEVAL
        # ----------------------------------------------------

        if not documents:

            answer = (
                "I don't know based on the provided "
                "VM0042 documents."
            )

            st.markdown(answer)

        else:

            # ------------------------------------------------
            # CREATE CONTEXT
            # ------------------------------------------------

            context = format_context(
                documents
            )


            # ------------------------------------------------
            # GENERATE ANSWER
            # ------------------------------------------------

            with st.spinner(
                "🤖 Generating answer..."
            ):

                try:

                    answer = generate_answer(
                        question,
                        context
                    )

                except Exception as e:

                    st.error(
                        "❌ Error while generating the answer."
                    )

                    st.exception(e)

                    answer = None


            # ------------------------------------------------
            # DISPLAY ANSWER
            # ------------------------------------------------

            if answer:

                st.markdown(answer)


            # ------------------------------------------------
            # SHOW SOURCES
            # ------------------------------------------------

            

    # --------------------------------------------------------
    # SAVE ASSISTANT RESPONSE
    # --------------------------------------------------------

    if answer:

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )
