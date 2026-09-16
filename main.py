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
# MODEL
# ============================================================

MODEL_NAME = "openai/gpt-oss-120b"


# ============================================================
# EMBEDDING MODEL
# ============================================================

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@st.cache_resource
def load_embeddings():

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
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

    st.error("❌ Vector store folder not found.")

    st.code(
        """
vm0042-agent/
│
├── main.py
├── requirements.txt
│
└── vector_store/
    ├── index.faiss
    └── index.pkl
"""
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

    st.write(
        "The FAISS index may have been created using a different "
        "embedding model."
    )

    st.write(
        f"Current embedding model: `{EMBEDDING_MODEL}`"
    )

    st.exception(e)

    st.stop()


# ============================================================
# CHECK EMBEDDING DIMENSION
# ============================================================

try:

    faiss_dimension = vector_store.index.d

    test_embedding = embeddings.embed_query(
        "VM0042 eligibility criteria"
    )

    embedding_dimension = len(test_embedding)

except Exception as e:

    st.error("❌ Could not check embedding dimensions.")
    st.exception(e)
    st.stop()


# ============================================================
# DIMENSION VALIDATION
# ============================================================

if faiss_dimension != embedding_dimension:

    st.error("❌ FAISS embedding dimension mismatch.")

    st.warning(
        f"""
The existing FAISS index expects:

**{faiss_dimension} dimensions**

But the current embedding model produces:

**{embedding_dimension} dimensions**
"""
    )

    st.markdown(
        f"""
### Current embedding model

`{EMBEDDING_MODEL}`

### What you need to do

The FAISS index must be rebuilt using the **same embedding model**
used by this application.

Do not simply change the model name unless you also rebuild the
FAISS index.
"""
    )

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
# PROMPT
# ============================================================

PROMPT_TEMPLATE = """
You are a technical AI assistant specialized in the Verra VM0042
Improved Agricultural Land Management methodology.

Your task is to answer the user's question using ONLY the information
provided in the VM0042 document context.

IMPORTANT RULES:

1. Use only the provided VM0042 context.

2. Do not use your own knowledge.

3. Do not use outside information.

4. Do not make assumptions.

5. Do not invent any information.

6. If the answer cannot be found in the context, respond exactly:

"I don't know based on the provided VM0042 documents."

7. Do not invent or modify:
- eligibility criteria
- applicability conditions
- project activities
- requirements
- definitions
- equations
- variables
- monitoring requirements
- emission reduction requirements
- baseline requirements
- leakage requirements
- project boundary requirements

8. For eligibility questions, provide only eligibility information
explicitly available in the context.

9. For project activity questions, provide only project activities
explicitly available in the context.

10. For applicability questions, provide only applicability conditions
explicitly available in the context.

11. If multiple document sections are relevant, combine them carefully.

12. If the context contains conflicting information, clearly mention
the conflict.

13. For equations:
- Use only equations found in the context.
- Explain the variables.
- Use only values provided in the context.
- Do not create equations.

14. Answer naturally and clearly.

15. Keep the answer concise.

16. Mention the document page or section when available.

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

            source_info = (
                f"{source}, page {page}"
            )

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

def generate_answer(
    question,
    context
):

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

    with st.spinner(
        "🔎 Searching VM0042 documents..."
    ):

        documents = retrieve_documents(
            question
        )


    # --------------------------------------------------------
    # CHECK DOCUMENTS
    # --------------------------------------------------------

    if not documents:

        st.warning(
            "I don't know based on the provided VM0042 documents."
        )

        st.stop()


    # --------------------------------------------------------
    # FORMAT CONTEXT
    # --------------------------------------------------------

    context = format_context(
        documents
    )


    # --------------------------------------------------------
    # GENERATE ANSWER
    # --------------------------------------------------------

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

            st.stop()


    # --------------------------------------------------------
    # DISPLAY ANSWER
    # --------------------------------------------------------

    st.subheader("🤖 Answer")

    st.write(answer)


    # --------------------------------------------------------
    # SOURCES
    # --------------------------------------------------------

    with st.expander(
        "📚 Retrieved VM0042 Sources"
    ):

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
