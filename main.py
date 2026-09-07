from pathlib import Path
import os
from dotenv import load_dotenv
from huggingface_hub import whoami
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

st.set_page_config(page_title="VM0042 Agent", page_icon="🤖")
st.title("VM0042 Agent")
st.write("Hi! Welcome to the VM0042 Question Answering System.")

BASE_DIR = Path(__file__).resolve().parent
VECTOR_STORE_DIR = BASE_DIR / "vector_store"
INDEX_PATH = VECTOR_STORE_DIR / "index.faiss"

if not INDEX_PATH.exists():
    st.error(f"FAISS index not found: {INDEX_PATH}")
    st.stop()


@st.cache_resource
def load_rag_pipeline():
    # 1. Retrieve & Validate API Token
    hf_token = st.secrets.get("HUGGINGFACEHUB_API_TOKEN") or os.getenv(
        "HUGGINGFACEHUB_API_TOKEN"
    )

    if not hf_token:
        st.error(
            "Hugging Face API Token missing! Please add HUGGINGFACEHUB_API_TOKEN to your .env or secrets.toml file."
        )
        st.stop()

    # Test token validity before initializing models
    try:
        user_info = whoami(token=hf_token)
        st.sidebar.success(
            f"Authenticated as: {user_info.get('name', 'User')}"
        )
    except Exception as token_err:
        st.error(
            f"Hugging Face Authentication Failed: {token_err}. Please generate a new token."
        )
        st.stop()

    os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token

    # 2. Embeddings (384 Dimensions)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 3. Vector Store
    vector_store = FAISS.load_local(
        folder_path=str(VECTOR_STORE_DIR),
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )

    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 4}
    )

    # 4. LLM Setup
    llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    task="text-generation",

)



    chat_model = ChatHuggingFace(llm=llm)

    return vector_store, retriever, chat_model


# Initialize models and vector store
try:
    vector_store, retriever, chat_model = load_rag_pipeline()

    st.sidebar.markdown("### Vector Store Metrics")
    st.sidebar.write("Total Vectors:", vector_store.index.ntotal)
    st.sidebar.write("Vector Dimension:", vector_store.index.d)

except Exception as e:
    st.error(f"Failed to initialize RAG pipeline: {e}")
    st.stop()

# Prompt & Chain
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


parallel_chain = RunnableParallel(
    {
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    }
)

parser = StrOutputParser()
main_chain = parallel_chain | prompt | chat_model | parser

question = st.text_input("Ask a question about VM0042:", key="user_question")

if question:
    with st.spinner("Generating answer..."):
        try:
            final_result = main_chain.invoke(question)
            st.write("### Answer")
            st.write(final_result)
        except Exception as e:
            st.error(f"Error generating answer: {e}")
