
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.output_parsers import StrOutputParser


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

st.set_page_config(
    page_title="VM0033 RAG Chatbot",
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 VM0042 Question Answering System")

st.caption(
    "AI assistant for the Verra VM0042 Tidal Wetland "
    "and Seagrass Restoration methodology."
)


# ============================================================
# GROQ API KEY
# ============================================================

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except (KeyError, FileNotFoundError):
        api_key = None

if not api_key:
    st.error(
        "GROQ_API_KEY is missing. Add it to your .env file "
        "or Streamlit Cloud Secrets."
    )
    st.stop()


# ============================================================
# VECTOR STORE PATH
# ============================================================

VECTOR_STORE_PATH = Path(__file__).parent / "vector_store"

FAISS_INDEX = VECTOR_STORE_PATH / "index.faiss"
FAISS_PICKLE = VECTOR_STORE_PATH / "index.pkl"


if not FAISS_INDEX.exists() or not FAISS_PICKLE.exists():
    st.error(
        "FAISS vector store files were not found.\n\n"
        "Expected files:\n"
        "- vector_store/index.faiss\n"
        "- vector_store/index.pkl"
    )
    st.stop()


# ============================================================
# LOAD EMBEDDINGS AND FAISS
# ============================================================

@st.cache_resource
def load_retriever():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.load_local(
        str(VECTOR_STORE_PATH),
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 6,
            "fetch_k": 30,
            "lambda_mult": 0.6
        }
    )


try:
    retriever = load_retriever()

except Exception as e:
    st.error("Failed to load the FAISS vector database.")
    st.exception(e)
    st.stop()


# ============================================================
# GROQ MODEL
# ============================================================

model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.4,
    api_key=api_key
)


# ============================================================
# PROMPT TEMPLATE - YOUR VM0033 PROMPT
# ============================================================

prompt = PromptTemplate(
    template="""
You are a technical assistant specialized in the Verra VM0042
Tidal Wetland and Seagrass Restoration methodology.

Answer the user's question using ONLY the information provided
in the context below.

IMPORTANT RULES:

1. Use only the provided context. Do not use outside knowledge
   or assumptions to fill missing information.

2. If the answer cannot be found in the context, respond exactly:
   "I don't know based on the provided VM0042 documents."

3. Do not invent, modify, or assume any methodology requirements,
   values, equations, variables, definitions, or eligibility criteria.

4. For questions about equations or calculations:
   - Identify the relevant equation from the context.
   - Write the equation clearly.
   - Explain each variable and parameter.
   - Substitute the given values.
   - Show the calculation step by step.
   - Give the final result with the correct unit.
   - Do not create an equation if it is absent from the context.

5. If multiple sections are relevant, combine them carefully.

6. If the context contains conflicting information, explain
   the conflict rather than making assumptions.

7. For eligibility, baseline, project boundaries, additionality,
   leakage, emission reductions, and monitoring, use the exact
   requirements provided in the context.

8. Keep the answer concise, factual, and easy to understand.

CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
""",
    input_variables=["context", "question"]
)


# ============================================================
# FORMAT RETRIEVED DOCUMENTS
# ============================================================

def format_docs(retrieved_docs):

    return "\n\n".join(
        doc.page_content for doc in retrieved_docs
    )


# ============================================================
# BUILD LANGCHAIN RAG CHAIN
# ============================================================

parallel_chain = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
})

parser = StrOutputParser()

main_chain = parallel_chain | prompt | model | parser


# ============================================================
# CHAT HISTORY
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🌱 VM0033 Agent")

    st.write(
        "Ask questions about the Verra VM0033 methodology."
    )

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ============================================================
# DISPLAY PREVIOUS MESSAGES
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

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        try:
            with st.spinner(
                "🔎 Searching VM0033 documents and generating answer..."
            ):

                # Retrieve documents once
                documents = retriever.invoke(question)

                if not documents or not any(
                    doc.page_content.strip() for doc in documents
                ):
                    answer = (
                        "I don't know based on the provided VM0033 documents."
                    )

                else:
                    # Build context from retrieved documents
                    context = format_docs(documents)

                    # Generate answer using the exact prompt
                    # and Groq model
                    answer = (
                        ({"context": lambda _: context,
                          "question": RunnablePassthrough()})
                        | prompt
                        | model
                        | parser
                    ).invoke(question)

            st.markdown(answer)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })

            # Optional: show retrieved sources
            with st.expander("View retrieved document chunks"):
                for i, doc in enumerate(documents, start=1):
                    st.markdown(f"**Document chunk {i}**")
                    st.write(doc.page_content[:1500])

                    if doc.metadata:
                        st.caption(
                            f"Metadata: {doc.metadata}"
                        )

        except Exception as e:
            st.error("Failed to generate the answer.")
            st.exception(e)
