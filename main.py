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
# PAGE CONFIGURATION
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
    "Verra VM0042 Improved Agricultural Land Management "
    "document-based AI assistant"
)


# ============================================================
# HUGGING FACE TOKEN
# ============================================================

try:
    HF_TOKEN = st.secrets["HF_TOKEN"]
except Exception:
    HF_TOKEN = os.getenv("HF_TOKEN")


if not HF_TOKEN:

    st.error(
        """
        ❌ Hugging Face token not found.

        Please add `HF_TOKEN` to:

        Streamlit Cloud → Manage app → Settings → Secrets
        """
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
# CHECK VECTOR STORE FOLDER
# ============================================================

if not VECTOR_STORE_PATH.exists():

    st.error(
        f"""
        ❌ Vector store folder not found.

        Expected:

        `{VECTOR_STORE_PATH}`

        Your GitHub repository should contain:

        vm0042-agent/
        ├── main.py
        ├── requirements.txt
        └── vector_store/
            ├── index.faiss
            └── index.pkl
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
        f"""
        ❌ `index.faiss` not found.

        Expected location:

        `{FAISS_INDEX}`
        """
    )

    st.stop()


if not FAISS_PICKLE.exists():

    st.error(
        f"""
        ❌ `index.pkl` not found.

        Expected location:

        `{FAISS_PICKLE}`
        """
    )

    st.stop()


# ============================================================
# LOAD FAISS VECTOR STORE
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
        """
        Make sure that:

        1. `index.faiss` exists
        2. `index.pkl` exists
        3. Both files were created using the same embedding model
        """
    )

    st.exception(e)

    st.stop()


# ============================================================
# CREATE RETRIEVER
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
# LOAD HUGGING FACE LLM
# ============================================================

@st.cache_resource
def load_llm():

    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
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

Your task is to answer the user's question using ONLY the
information provided in the context below.

IMPORTANT RULES:

1. Use ONLY the provided context.

2. Do not use your own knowledge, assumptions, or information
outside the provided VM0042 documents.

3. If the answer cannot be found in the context, respond exactly:

"I don't know based on the provided VM0042 documents."

4. Do not invent, modify, or assume any VM0042:
   - requirements
   - values
   - equations
   - variables
   - definitions
   - eligibility criteria
   - procedures
   - monitoring requirements

5. For questions about equations or calculations:

   - Identify the relevant equation from the context.
   - Write the equation clearly.
   - Explain each variable.
   - Substitute the provided values.
   - Show the calculation step by step.
   - Give the final result with the correct unit.
   - Do not create an equation that is not present in the context.

6. If multiple sections of the context are relevant, combine
them carefully into one clear answer.

7. If the context contains conflicting information, mention the
conflict instead of choosing an answer based on assumption.

8. For questions about:
   - project activity
   - applicability
   - eligibility
   - baseline
   - project boundary
   - additionality
   - leakage
   - emission reductions
   - monitoring

   use the exact information available in the context.

9. Keep the answer concise, factual, and easy to understand.

10. When possible, mention the relevant section, equation,
document information, or page information available in the
context.

------------------------------------------------------------
CONTEXT
------------------------------------------------------------

{context}

------------------------------------------------------------
USER QUESTION
------------------------------------------------------------

{question}

------------------------------------------------------------
ANSWER
------------------------------------------------------------

""",
    input_variables=[
        "context",
        "question"
    ]
)


# ============================================================
# FORMAT RETRIEVED DOCUMENTS
# ============================================================

def format_docs(retrieved_docs):

    if not retrieved_docs:

        return (
            "No relevant VM0042 documents were found."
        )

    context_parts = []

    for doc in retrieved_docs:

        context_parts.append(
            doc.page_content
        )

    return "\n\n--- DOCUMENT SECTION ---\n\n".join(
        context_parts
    )


# ============================================================
# PARALLEL RETRIEVAL CHAIN
# ============================================================

parallel_chain = RunnableParallel(
    {
        "context": (
            retriever
            | RunnableLambda(format_docs)
        ),

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
# SESSION STATE
# ============================================================

if "qa_history" not in st.session_state:

    st.session_state.qa_history = []


# ============================================================
# CREATE TABS
# ============================================================

chat_tab, history_tab = st.tabs(
    [
        "💬 Ask Questions",
        "📚 Question & Answer History"
    ]
)


# ============================================================
# CHATBOT TAB
# ============================================================

with chat_tab:

    st.subheader(
        "🔎 Ask a question about VM0042"
    )

    question = st.text_input(
        "Enter your question:",
        placeholder=(
            "Example: What is the applicability "
            "of VM0042?"
        ),
        key="question_input"
    )


    # --------------------------------------------------------
    # ASK BUTTON
    # --------------------------------------------------------

    ask_button = st.button(
        "🔍 Ask Question",
        type="primary"
    )


    if ask_button:

        if not question.strip():

            st.warning(
                "⚠️ Please enter a question."
            )

        else:

            with st.spinner(
                "🔎 Searching VM0042 documents..."
            ):

                try:

                    # Generate answer
                    answer = main_chain.invoke(
                        question
                    )


                    # ----------------------------------------
                    # DISPLAY ANSWER
                    # ----------------------------------------

                    st.subheader(
                        "🤖 Answer"
                    )

                    st.write(answer)


                    # ----------------------------------------
                    # SAVE QUESTION + ANSWER
                    # ----------------------------------------

                    st.session_state.qa_history.append(
                        {
                            "question": question,
                            "answer": answer
                        }
                    )


                except Exception as e:

                    st.error(
                        "❌ Error while generating the answer."
                    )

                    st.exception(e)


# ============================================================
# QUESTION & ANSWER HISTORY TAB
# ============================================================

with history_tab:

    st.subheader(
        "📚 All Asked Questions and Answers"
    )


    # --------------------------------------------------------
    # NO HISTORY
    # --------------------------------------------------------

    if not st.session_state.qa_history:

        st.info(
            "No questions have been asked yet."
        )


    # --------------------------------------------------------
    # DISPLAY HISTORY
    # --------------------------------------------------------

    else:

        st.write(
            f"Total questions: "
            f"**{len(st.session_state.qa_history)}**"
        )


        # Newest question first
        for i, item in enumerate(
            reversed(
                st.session_state.qa_history
            ),
            1
        ):

            question_number = (
                len(st.session_state.qa_history)
                - i
                + 1
            )


            st.markdown(
                f"### Question {question_number}"
            )


            st.markdown(
                "**❓ Question:**"
            )

            st.write(
                item["question"]
            )


            st.markdown(
                "**🤖 Answer:**"
            )

            st.write(
                item["answer"]
            )


            st.divider()


        # ----------------------------------------------------
        # CLEAR HISTORY
        # ----------------------------------------------------

        if st.button(
            "🗑️ Clear Question History"
        ):

            st.session_state.qa_history = []

            st.rerun()
