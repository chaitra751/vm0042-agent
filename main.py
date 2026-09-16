import os
from pathlib import Path
import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="VM0042 Agent",
    page_icon="🌱",
    layout="centered"
)

st.title("🌱 VM0042 Question Answering System")
st.caption("AI assistant for the Verra VM0042 Improved Agricultural Land Management methodology")


# ============================================================
# HUGGING FACE TOKEN & SECRETS CHECK
# ============================================================

HF_TOKEN = st.secrets.get("HF_TOKEN")

if not HF_TOKEN:
    st.error("❌ Missing `HF_TOKEN` in Streamlit Secrets.")
    st.info("Go to Streamlit Cloud → App → Settings → Secrets and add `HF_TOKEN`.")
    st.stop()

HF_TOKEN = HF_TOKEN.strip()


# ============================================================
# INITIALIZE RAG COMPONENTS (CACHED)
# ============================================================

VECTOR_STORE_PATH = Path(__file__).parent / "vector_store"
MODEL_NAME = "openai/gpt-oss-120b"


@st.cache_resource
def init_rag_components():
    # 1. Check Vector Store Path
    if not VECTOR_STORE_PATH.exists():
        st.error(f"❌ Vector store folder not found at: `{VECTOR_STORE_PATH}`")
        st.stop()

    # 2. Embeddings Model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 3. Load FAISS Store
    vector_store = FAISS.load_local(
        str(VECTOR_STORE_PATH),
        embeddings,
        allow_dangerous_deserialization=True
    )

    # 4. Create Retriever
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 6,
            "fetch_k": 30,
            "lambda_mult": 0.6
        }
    )

    chat_model = ChatOpenAI(
    model="openai/gpt-oss-120b",
    openai_api_key=HF_TOKEN,
    openai_api_base="https://api-inference.huggingface.co/models/openai/gpt-oss-120b/v1",
    temperature=0.1,
    max_tokens=512,
)

    return retriever, chat_model


try:
    retriever, chat_model = init_rag_components()
except Exception as e:
    st.error("❌ Failed to initialize system components.")
    st.exception(e)
    st.stop()


# ============================================================
# PROMPT TEMPLATE
# ============================================================

PROMPT_TEMPLATE = """
You are a technical AI assistant specialized in the Verra VM0042
Improved Agricultural Land Management methodology.

Your task is to answer the user's question using ONLY the information
provided in the VM0042 document context.

IMPORTANT RULES:
1. Use only the provided context. Do not use outside knowledge.
2. If the answer cannot be found in the context, respond exactly:
   "I don't know based on the provided VM0042 documents."
3. Do not invent, modify, or assume any VM0042 requirements, values, or equations.
4. Keep the answer concise, factual, and natural.

--------------------------------------------------
VM0042 DOCUMENT CONTEXT
--------------------------------------------------
{context}

--------------------------------------------------
USER QUESTION
--------------------------------------------------
{question}
"""

prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)


# ============================================================
# FORMAT RETRIEVED DOCUMENTS
# ============================================================

def format_docs(retrieved_docs):
    if not retrieved_docs:
        return "No relevant VM0042 documents were found."

    context_parts = []
    for i, doc in enumerate(retrieved_docs, start=1):
        metadata = doc.metadata or {}
        source = metadata.get("source", "VM0042 document")
        page = metadata.get("page", metadata.get("page_number", ""))
        source_info = f"{source}, page {page}" if page != "" else str(source)
        
        context_parts.append(
            f"--- DOCUMENT {i} ({source_info}) ---\n{doc.page_content}"
        )

    return "\n\n".join(context_parts)


# ============================================================
# LCEL RAG PIPELINE
# ============================================================

parallel_chain = RunnableParallel(
    {
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    }
)

main_chain = parallel_chain | prompt | chat_model | StrOutputParser()


# ============================================================
# USER INPUT & EXECUTION INTERFACE
# ============================================================

with st.form("qa_form"):
    question = st.text_input(
        "🔎 Ask your question about VM0042",
        placeholder="Example: What is the applicability of VM0042?"
    )
    submit_button = st.form_submit_button("Get Answer", type="primary")


if submit_button and question.strip():
    with st.spinner("🔎 Searching VM0042 documents & generating answer..."):
        try:
            # Execute LCEL Chain
            answer = main_chain.invoke(question)

            # Render Answer
            st.subheader("🤖 Answer")
            st.write(answer)

            # Render Referenced Sources in Expander
            retrieved_docs = retriever.invoke(question)
            if retrieved_docs:
                with st.expander("📚 View Referenced Sources"):
                    for i, doc in enumerate(retrieved_docs, start=1):
                        metadata = doc.metadata or {}
                        source = metadata.get("source", "VM0042 document")
                        page = metadata.get("page", metadata.get("page_number", ""))
                        page_str = f" — Page {page}" if page != "" else ""

                        st.markdown(f"**Document {i}:** {source}{page_str}")
                        st.caption(doc.page_content[:250] + "...")

        except Exception as e:
            st.error("❌ Error while generating the answer.")
            st.exception(e)
