import os
from pathlib import Path

import streamlit as st

from langchain_huggingface import (
    HuggingFaceEmbeddings,
    HuggingFaceEndpoint,
    ChatHuggingFace
)

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

# ============================================================
# HUGGING FACE TOKEN
# ============================================================

try:
    HF_TOKEN = st.secrets["HF_TOKEN"]
except Exception:
    HF_TOKEN = os.getenv("HF_TOKEN")


if not HF_TOKEN:
    st.error(
        "❌ Hugging Face token not found.\n\n"
        "Add HF_TOKEN to Streamlit Cloud → Manage app → Settings → Secrets."
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

        ```
        vm0042-agent/
        ├── main.py
        ├── requirements.txt
        └── vector_store/
            ├── index.faiss
            └── index.pkl
        ```
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
# LOAD FAISS
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
        "with the same embedding model."
    )
    st.exception(e)
    st.stop()


# ============================================================
# CREATE RETRIEVER
# ============================================================

#retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 6,
        "fetch_k": 30,
        "lambda_mult": 0.6
    }
)


# ============================================================
# HUGGING FACE LLM
# ============================================================

@st.cache_resource
def load_llm():

    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.3-70B-Instruct-evals",
        task="text-generation",
        max_new_tokens=512,
        temperature=0.1,
        huggingfacehub_api_token=HF_TOKEN
    )

    chat_model = ChatHuggingFace(
        llm=llm
    )

    return chat_model


try:

    chat_model = load_llm()

except Exception as e:

    st.error("❌ Failed to initialize Hugging Face LLM.")
    st.exception(e)
    st.stop()


# ============================================================
# PROMPT
# ============================================================

prompt = PromptTemplate(
    template="""
You are a technical assistant specialized in the Verra VM0042
Improved Agricultural Land Management methodology.

Your task is to answer the user's question using ONLY the information
provided in the context below.

IMPORTANT RULES:

1. Use only the provided context. Do not use your own knowledge or
   assumptions to fill missing information.

2. If the answer cannot be found in the context, respond exactly:
   "I don't know based on the provided VM0042 documents."

3. Do not invent, modify, or assume any VM0042 requirements, values,
   equations, variables, definitions, or eligibility criteria.

4. For questions about equations or calculations:
   - First identify the relevant equation from the context.
   - Write the equation clearly.
   - Explain each variable and parameter.
   - Substitute the given values.
   - Show the calculation step by step.
   - Give the final result with the correct unit.
   - Do not create an equation if it is not present in the context.

5. If multiple sections of the context are relevant, combine them
   carefully and provide one clear answer.

6. If the context contains conflicting information, mention the
   conflict instead of choosing an answer by assumption.

7. For questions about project activity eligibility, baseline,
   project boundaries, additionality, leakage, emission reductions,
   or monitoring, use the exact requirements available in the context.

8. Keep the answer concise, factual, and easy to understand.

9. When possible, mention the relevant document section, equation,
   or page information available in the context.


USER QUESTION:
{question}

ANSWER:
""",
    input_variables=["context", "question"]
)


# ============================================================
# FORMAT DOCUMENTS
# ============================================================

def format_docs(retrieved_docs):

    if not retrieved_docs:
        return "No relevant documents were found."

    context_text = "\n\n".join(
        doc.page_content
        for doc in retrieved_docs
    )

    return context_text


# ============================================================
# PARALLEL RETRIEVAL CHAIN
# ============================================================

parallel_chain = RunnableParallel(
    {
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    }
)


# ============================================================
# OUTPUT PARSER
# ============================================================

parser = StrOutputParser()

# ============================================================
# MAIN RAG CHAIN
# ============================================================

main_chain = (
    parallel_chain
    | prompt
    | chat_model
    | parser
)

# ============================================================
# USER QUESTION
# ============================================================

question = st.text_input(
    "🔎 Ask your question about VM0042",
    placeholder="Example: What is the applicability of VM0042?"
)

if question:

    with st.spinner("🔎 Searching VM0042 documents..."):

        try:

            answer = main_chain.invoke(question)

            st.subheader("🤖 Answer")

            st.write(answer)

        except Exception as e:

            st.error(
                "❌ Error while generating the answer."
            )

            st.exception(e)

