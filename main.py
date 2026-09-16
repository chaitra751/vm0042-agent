```python
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
    st.stop()

HF_TOKEN = HF_TOKEN.strip()


# ============================================================
# LOAD EMBEDDING MODEL
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

VECTOR_STORE_PATH = Path(__file__).parent / "vector_store"

FAISS_INDEX = VECTOR_STORE_PATH / "index.faiss"
FAISS_PICKLE = VECTOR_STORE_PATH / "index.pkl"


if not VECTOR_STORE_PATH.exists():

    st.error("❌ vector_store folder not found.")
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

    st.error("❌ Failed to load FAISS vector store.")

    st.write(
        "Your FAISS index must be created using:"
    )

    st.code(
        "sentence-transformers/all-MiniLM-L6-v2"
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

PROMPT = """
You are a technical assistant specialized in the Verra VM0042
Improved Agricultural Land Management methodology.

Answer the question ONLY using the provided VM0042 document context.

Rules:

1. Use only the information in the context.
2. Do not use outside knowledge.
3. Do not make assumptions.
4. Do not invent requirements, values, definitions, equations,
   eligibility criteria, project activities, or monitoring rules.
5. If the answer is not available in the context, say:

"I don't know based on the provided VM0042 documents."

6. For eligibility questions, list only the eligibility criteria
   found in the context.

7. For project activity questions, list only the activities found
   in the context.

8. For applicability questions, use only the applicability
   conditions found in the context.

9. For equations, use only equations explicitly available in
   the context.

10. Answer naturally and clearly.

11. Keep the answer concise.

12. Mention the document section or page when available.

------------------------------------------------------------
VM0042 DOCUMENT CONTEXT
------------------------------------------------------------

{context}

------------------------------------------------------------
QUESTION
------------------------------------------------------------

{question}

------------------------------------------------------------
ANSWER
------------------------------------------------------------
"""


# ============================================================
# RETRIEVE DOCUMENTS
# ============================================================

def retrieve_documents(question):

    try:

        documents = retriever.invoke(question)

        return documents

    except Exception as e:

        st.error("❌ Error retrieving documents.")
        st.exception(e)

        return []


# ============================================================
# FORMAT DOCUMENT CONTEXT
# ============================================================

def format_context(documents):

    if not documents:

        return "No relevant documents found."

    context = []

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

            source_info = source

        context.append(
            f"""
DOCUMENT {i}
Source: {source_info}

{doc.page_content}
"""
        )

    return "\n\n".join(context)


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(question, context):

    final_prompt = PROMPT.format(
        context=context,
        question=question
    )

    response = client.chat.completions.create(

        model=MODEL_NAME,

        messages=[
            {
                "role": "user",
                "content": final_prompt
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
# QUESTION ANSWER
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
    # FORMAT CONTEXT
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

    with st.expander("📚 Sources"):

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
                    f"Document {i}: {source} — Page {page}"
                )

            else:

                st.write(
                    f"Document {i}: {source}"
                )
```
