from pathlib import Path
import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI  # Or your preferred chat model class

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="VM0042 Agent", page_icon="🌱")
st.title("🌱 VM0042 Question Answering System")

# ============================================================
# INITIALIZE VECTOR STORE & LLM
# ============================================================
VECTOR_STORE_PATH = Path(__file__).parent / "vector_store"


@st.cache_resource
def init_rag_components():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.load_local(
        str(VECTOR_STORE_PATH),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 30, "lambda_mult": 0.6},
    )

    # Replace with your initialized Chat Model
    chat_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

    return retriever, chat_model


try:
    retriever, chat_model = init_rag_components()
except Exception as e:
    st.error("❌ Failed to initialize RAG components.")
    st.exception(e)
    st.stop()

# ============================================================
# PROMPT TEMPLATE
# ============================================================
prompt = ChatPromptTemplate.from_template(
    """You are an assistant specialized in the Verra VM0042 methodology.
Answer the question using ONLY the provided context. If you do not know, say "I don't know based on the provided VM0042 documents."

Context:
{context}

Question:
{question}
"""
)

# ============================================================
# RAG PIPELINE SETUP
# ============================================================


def format_docs(retrieved_docs):
    if not retrieved_docs:
        return "No relevant documents were found."
    return "\n\n".join(doc.page_content for doc in retrieved_docs)


parallel_chain = RunnableParallel(
    {
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    }
)

main_chain = parallel_chain | prompt | chat_model | StrOutputParser()

# ============================================================
# USER INPUT & OUTPUT
# ============================================================
with st.form("qa_form"):
    question = st.text_input(
        "🔎 Ask your question about VM0042",
        placeholder="Example: What is the applicability of VM0042?",
    )
    submit = st.form_submit_button("Submit", type="primary")

if submit and question.strip():
    with st.spinner("🔎 Searching VM0042 documents..."):
        try:
            answer = main_chain.invoke(question)
            st.subheader("🤖 Answer")
            st.write(answer)
        except Exception as e:
            st.error("❌ Error while generating the answer.")
            st.exception(e)
