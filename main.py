import streamlit as st
from pathlib import Path

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

FAISS_INDEX = VECTOR_STORE_PATH / "index.faiss"
FAISS_PICKLE = VECTOR_STORE_PATH / "index.pkl"


# ============================================================
# CHECK VECTOR STORE
# ============================================================

if not VECTOR_STORE_PATH.exists():

    st.error(
        f"""
❌ Vector store folder not found.

Expected location:

`{VECTOR_STORE_PATH}`

Your repository should contain:

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
        f"❌ `index.faiss` not found inside `{VECTOR_STORE_PATH}`"
    )

    st.stop()


if not FAISS_PICKLE.exists():

    st.error(
        f"❌ `index.pkl` not found inside `{VECTOR_STORE_PATH}`"
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

    st.write(
        "Make sure the FAISS index was created using:"
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
# HUGGING FACE CLIENT
# ============================================================

@st.cache_resource
def load_llm():

    return InferenceClient(
        token=HF_TOKEN
    )


try:

    client = load_llm()

except Exception as e:

    st.error("❌ Failed to initialize Hugging Face client.")
    st.exception(e)
    st.stop()


# ============================================================
# SAME MODEL
# ============================================================

MODEL_NAME = "openai/gpt-oss-120b"


# ============================================================
# PROMPT
# ============================================================

PROMPT_TEMPLATE = """
You are a technical AI assistant specialized in the Verra VM0042
Improved Agricultural Land Management methodology.

Answer the user's question using ONLY the information provided
in the VM0042 document context.

IMPORTANT RULES:

1. Use only the provided context.

2. Do not use your own knowledge or outside information.

3. Do not make assumptions.

4. If the answer cannot be found in the context, respond exactly:

"I don't know based on the provided VM0042 documents."

5. Do not invent or modify VM0042:
- requirements
- values
- equations
- variables
- definitions
- eligibility criteria
- project activities
- monitoring requirements

6. For equations or calculations:
- Use only equations present in the context.
- Explain the variables.
- Substitute the provided values.
- Show the calculation.
- Give the final result with the correct unit.

7. If multiple sections are relevant, combine them carefully.

8. If the context contains conflicting information, mention the conflict.

9. For eligibility, applicability, baseline, project boundaries,
additionality, leakage, emission reductions, monitoring, or project
activities, use only requirements explicitly available in the context.

10. Answer naturally and clearly.

11. Do not copy large sections of the document.

12. Keep the answer concise and factual.

13. Mention the document section or page when available.

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

        return retriever.invoke(question)

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
# QUESTION INPUT
# ============================================================

question = st.text_input(
    "🔎 Ask your question about VM0042",
    placeholder="Example: What are the eligibility criteria?"
)


# ============================================================
# PROCESS QUESTION
# ============================================================

if question:

    # --------------------------------------------------------
    # RETRIEVE
    # --------------------------------------------------------

    with st.spinner("🔎 Searching VM0042 documents..."):

        documents = retrieve_documents(question)


    # --------------------------------------------------------
    # NO DOCUMENTS
    # --------------------------------------------------------

    if not documents:

        st.warning(
            "I don't know based on the provided VM0042 documents."
        )

        st.stop()


    # --------------------------------------------------------
    # CREATE CONTEXT
    # --------------------------------------------------------

    context = format_context(documents)


    # --------------------------------------------------------
    # GENERATE ANSWER
    # --------------------------------------------------------

    with st.spinner("🤖 Generating answer..."):

        try:

            answer = generate_answer(
                question,
                context
            )

        except Exception as e:

            st.error("❌ Error while generating the answer.")
            st.exception(e)

            st.stop()


    # --------------------------------------------------------
    # DISPLAY ANSWER
    # --------------------------------------------------------

    st.subheader("🤖 Answer")

    st.write(answer)


    # --------------------------------------------------------
    # SOURCES
    # --------------------------------------------------------

    with st.expander("📚 Retrieved VM0042 Sources"):

        for i, doc in enumerate(
            documents,
            start=1
        ):

            metadata = doc.metadata or {}

            source = metadata.get(
                "source",
                "VM0042 document"
            )

            page = metadata.get(
                "page",
                metadata.get(
                    "page_number",
                    ""
                )
            )

            if page != "":

                st.markdown(
                    f"**Document {i}:** "
                    f"{source} — Page {page}"
                )

            else:

                st.markdown(
                    f"**Document {i}:** "
                    f"{source}"
                )
