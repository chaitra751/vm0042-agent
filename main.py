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
    "Ask questions about the Verra VM0042 Improved Agricultural Land Management methodology."
)


# ============================================================
# HUGGING FACE TOKEN
# ============================================================

HF_TOKEN = st.secrets.get("HF_TOKEN")

if not HF_TOKEN:

    st.error("❌ HF_TOKEN is missing from Streamlit Secrets.")

    st.info(
        "Add HF_TOKEN in Streamlit Cloud → Settings → Secrets."
    )

    st.stop()

HF_TOKEN = HF_TOKEN.strip()


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

@st.cache_resource
def load_embeddings():

    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={
            "device": "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True
        }
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
        "❌ Vector store folder not found."
    )

    st.stop()


FAISS_INDEX = VECTOR_STORE_PATH / "index.faiss"
FAISS_PICKLE = VECTOR_STORE_PATH / "index.pkl"


if not FAISS_INDEX.exists():

    st.error(
        "❌ index.faiss not found."
    )

    st.stop()


if not FAISS_PICKLE.exists():

    st.error(
        "❌ index.pkl not found."
    )

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

    st.error(
        "❌ Unable to load FAISS vector store."
    )

    st.write(
        "The FAISS index must be created using the same embedding model:"
    )

    st.code(
        "BAAI/bge-base-en-v1.5"
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

    st.error(
        "❌ Failed to initialize Hugging Face client."
    )

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

Answer the user's question using ONLY the provided VM0042 document
context.

IMPORTANT RULES:

1. Use ONLY the provided context.

2. Do not use outside knowledge.

3. Do not make assumptions.

4. Do not invent information.

5. If the answer cannot be found in the context, respond exactly:

"I don't know based on the provided VM0042 documents."

6. For eligibility questions, provide only eligibility requirements
explicitly stated in the context.

7. For project activity questions, provide only project activities
explicitly stated in the context.

8. For applicability questions, provide only applicability conditions
explicitly stated in the context.

9. For equations:
   - Use only equations present in the context.
   - Explain the variables.
   - Show calculations only when values are provided.
   - Do not create equations.

10. If multiple document sections are relevant, combine them into
one clear answer.

11. If the documents contain conflicting information, mention the
conflict instead of making an assumption.

12. Answer naturally like a chatbot.

13. Keep the answer concise and factual.

14. Mention the relevant document section or page when available.

--------------------------------------------------
VM0042 DOCUMENT CONTEXT
--------------------------------------------------

{context}

--------------------------------------------------
QUESTION
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

        st.error(
            "❌ Error retrieving VM0042 documents."
        )

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
    "🔎 Ask your question",
    placeholder="Example: What are the eligibility criteria under VM0042?"
)


# ============================================================
# ANSWER
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


    if not documents:

        st.warning(
            "I don't know based on the provided VM0042 documents."
        )

        st.stop()


    # --------------------------------------------------------
    # CONTEXT
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

                st.write(
                    f"**Document {i}:** "
                    f"{source} — Page {page}"
                )

            else:

                st.write(
                    f"**Document {i}:** "
                    f"{source}"
                )
