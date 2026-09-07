import streamlit as st
from pathlib import Path
import base64

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="VM0042 Agent",
    page_icon="🌱",
    layout="wide"
)

# ==========================================
# TITLE
# ==========================================
st.title("🌱 VM0042 Question Answering System")

# ==========================================
# DATA FOLDER
# ==========================================
PDF_FOLDER = Path(__file__).parent / "data"

# Get PDF files
pdf_files = sorted(PDF_FOLDER.glob("*.pdf"))

# ==========================================
# SIDEBAR - PDF SELECTION
# ==========================================
st.sidebar.title("📚 VM0042 Documents")

if not pdf_files:
    st.sidebar.error("No PDF files found in data folder.")
    st.stop()

selected_pdf = st.sidebar.selectbox(
    "Select VM0042 Document",
    pdf_files,
    format_func=lambda x: x.name
)

st.sidebar.success(f"{len(pdf_files)} PDFs available")


# ==========================================
# TWO COLUMN LAYOUT
# ==========================================
left_col, right_col = st.columns([1, 1])


# ==========================================
# LEFT SIDE - PDF VIEWER
# ==========================================
with left_col:

    st.subheader("📄 Document")

    st.caption(selected_pdf.name)

    # Read PDF
    with open(selected_pdf, "rb") as pdf_file:
        pdf_bytes = pdf_file.read()

    # Convert PDF to base64
    base64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")

    # Display actual PDF
    pdf_display = f"""
    <iframe
        src="data:application/pdf;base64,{base64_pdf}"
        width="100%"
        height="750"
        style="border: 1px solid #ddd; border-radius: 8px;"
    >
    </iframe>
    """

    st.markdown(pdf_display, unsafe_allow_html=True)


# ==========================================
# RIGHT SIDE - CHATBOT
# ==========================================
with right_col:

    st.subheader("🤖 VM0042 Chatbot")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous messages
    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    user_question = st.chat_input(
        "Ask a question about VM0042..."
    )

    if user_question:

        # Display user question
        with st.chat_message("user"):
            st.write(user_question)

        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_question
            }
        )

        # --------------------------------------
        # TEMPORARY RESPONSE
        # Replace this with FAISS + Hugging Face
        # --------------------------------------

        answer = (
            "I received your question. "
            "The VM0042 FAISS + Hugging Face "
            "QA pipeline will generate the answer here."
        )

        with st.chat_message("assistant"):
            st.write(answer)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )
