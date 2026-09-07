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

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate


# =========================================================
# 1. EMBEDDINGS
# =========================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# =========================================================
# 2. LOAD FAISS VECTOR STORE
# =========================================================

vector_store = FAISS.load_local(
    folder_path="/content/vectore_store",
    embeddings=embeddings,
    allow_dangerous_deserialization=True
)


# =========================================================
# 3. CREATE RETRIEVER
# =========================================================

retriever = vector_store.as_retriever(
    search_kwargs={"k": 4}
)


# =========================================================
# 4. HUGGING FACE LLM
# =========================================================

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.1
)

chat_model = ChatHuggingFace(
    llm=llm
)


# =========================================================
# 5. PROMPT
# =========================================================

prompt = PromptTemplate(
    template="""
You are a helpful assistant for the VM0042
Improved Agricultural Land Management methodology.

Answer the user's question using only the provided context.

If the answer cannot be found in the context, say:
"I could not find this information in the VM0042 documents."

Context:
{context}

Question:
{question}

Answer:
""",
    input_variables=["context", "question"]
)


# =========================================================
# 6. FORMAT RETRIEVED DOCUMENTS
# =========================================================

def format_docs(retrieved_docs):

    context_text = "\n\n".join(
        doc.page_content
        for doc in retrieved_docs
    )

    return context_text


# =========================================================
# 7. PARALLEL RETRIEVAL CHAIN
# =========================================================

parallel_chain = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
})


# =========================================================
# 8. OUTPUT PARSER
# =========================================================

parser = StrOutputParser()


# =========================================================
# 9. MAIN RAG CHAIN
# =========================================================

main_chain = (
    parallel_chain
    | prompt
    | chat_model
    | parser
)


# =========================================================
# 10. STREAMLIT UI
# =========================================================

st.title("🌱 VM0042 Question Answering System")

question = st.text_input(
    "Ask a question about VM0042:"
)


if question:

    with st.spinner("Searching VM0042 documents..."):

        answer = main_chain.invoke(question)

    st.subheader("Answer")

    st.write(answer)
