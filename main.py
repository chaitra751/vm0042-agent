from pathlib import Path
import streamlit as st

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import (
    HuggingFaceEmbeddings,
    HuggingFaceEndpoint,
    ChatHuggingFace
)


st.title("VM0042 Agent")
st.write("Hi! Welcome to the VM0042 Question Answering System.")

BASE_DIR = Path(__file__).resolve().parent
VECTOR_STORE_DIR = BASE_DIR / "vector_store"
INDEX_PATH = VECTOR_STORE_DIR / "index.faiss"

if not INDEX_PATH.exists():
    st.error(f"FAISS index not found: {INDEX_PATH}")
    st.stop()

# Hugging Face Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load FAISS vector store
try:
    vector_store = FAISS.load_local(
        folder_path=str(VECTOR_STORE_DIR),
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )

    st.write("Number of vectors:", vector_store.index.ntotal)
    st.write("Vector dimension:", vector_store.index.d)

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    st.success("Vector store loaded successfully!")

except Exception as e:
    st.error(f"Failed to load vector store: {e}")
    st.stop()


hf_token = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=hf_token,
)

chat_model = ChatHuggingFace(llm=llm)


# User input
question = st.text_input("Ask a question about VM0042:")

prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)

question = "'What is vm0042"
retrieved_docs = retriever.invoke(question)

from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text

parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})

parser = StrOutputParser()
main_chain = parallel_chain | prompt | chat_model | parser



final_result=main_chain.invoke('What is the current version of VM0042?')
st.write(final_result)
