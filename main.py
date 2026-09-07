from pathlib import Path
from dotenv import load_dotenv
import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEmbeddings,
    HuggingFaceEndpoint,
)

load_dotenv()

st.title("VM0042 Agent")
st.write("Hi! Welcome to the VM0042 Question Answering System.")

BASE_DIR = Path(__file__).resolve().parent
VECTOR_STORE_DIR = BASE_DIR / "vector_store"
INDEX_PATH = VECTOR_STORE_DIR / "index.faiss"

if not INDEX_PATH.exists():
    st.error(f"FAISS index not found: {INDEX_PATH}")
    st.stop()


# ==========================================
# CACHED RESOURCE INITIALIZATION
# ==========================================
@st.cache_resource
def load_rag_pipeline():
    # 1. Hugging Face Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 2. Load FAISS vector store
    vector_store = FAISS.load_local(
        folder_path=str(VECTOR_STORE_DIR),
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )

    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 4}
    )

    # 3. Initialize Hugging Face LLM
    # Assumes HUGGINGFACEHUB_API_TOKEN is stored in st.secrets or .env
    hf_token = st.secrets.get(
        "HUGGINGFACEHUB_API_TOKEN", st.secrets.get("HF_TOKEN")
    )

    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        task="text-generation",
        max_new_tokens=512,
        temperature=0.1,
        huggingfacehub_api_token=hf_token,
    )

    chat_model = ChatHuggingFace(llm=llm)

    return vector_store, retriever, chat_model


try:
    vector_store, retriever, chat_model = load_rag_pipeline()

    # Display basic metrics once loaded
    st.sidebar.markdown("### Vector Store Metrics")
    st.sidebar.write("Total Vectors:", vector_store.index.ntotal)
    st.sidebar.write("Vector Dimension:", vector_store.index.d)

except Exception as e:
    st.error(f"Failed to initialize RAG pipeline: {e}")
    st.stop()


# ==========================================
# PROMPT TEMPLATE
# ==========================================

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer ONLY from the provided VM0042 context. If the context is insufficient, respond strictly with 'I don't know.'",
        ),
        (
            "human",
            """Context:
{context}

Question:
{question}""",
        ),
    ]
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# ==========================================
# CHAIN SETUP
# ==========================================

parallel_chain = RunnableParallel(
    {
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    }
)

parser = StrOutputParser()
main_chain = parallel_chain | prompt | chat_model | parser


# ==========================================
# USER QUESTION & INVOCATION
# ==========================================

question = st.text_input("Ask a question about VM0042:", key="user_question")

if question:
    with st.spinner("Generating answer..."):
        try:
            final_result = main_chain.invoke(question)
            st.write("### Answer")
            st.write(final_result)
        except Exception as e:
            st.error(f"Error generating answer: {e}")
