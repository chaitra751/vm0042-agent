import streamlit as st
from pathlib import Path
import base64
import streamlit.components.v1 as components

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="VM0042 Agent",
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 VM0042 Question Answering System")

# ==========================================
# DATA FOLDER
# ==========================================
PDF_FOLDER = Path(__file__).parent / "data"

pdf_files = sorted(PDF_FOLDER.glob("*.pdf"))

# ==========================================
# SIDEBAR
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
# TWO COLUMNS
# ==========================================
left_col, right_col = st.columns([1, 1])

# ==========================================
# LEFT SIDE - ACTUAL PDF
# ==========================================
with left_col:

    st.subheader("📄 VM0042 Document")
    st.caption(selected_pdf.name)

    # Read selected PDF
    with open(selected_pdf, "rb") as file:
        pdf_bytes = file.read()

    # Convert PDF to Base64
    pdf_base64 = base64.b64encode(pdf_bytes).decode("utf-8")

    # PDF Viewer
    pdf_html = f"""
    <html>
    <body style="margin:0; padding:0;">

        <iframe
            src="data:application/pdf;base64,{pdf_base64}"
            width="100%"
            height="750px"
            style="
                border: 1px solid #cccccc;
                border-radius: 8px;
            ">
        </iframe>

    </body>
    </html>
    """

    components.html(
        pdf_html,
        height=760,
        scrolling=False
    )


# ==========================================
# RIGHT SIDE - CHATBOT
# ==========================================
with right_col:

    st.subheader("🤖 VM0042 Chatbot")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    user_question = st.chat_input(
        "Ask a question about VM0042..."
    )

    if user_question:

        with st.chat_message("user"):
            st.write(user_question)

        st.session_state.messages.append({
            "role": "user",
            "content": user_question
        })

        # Temporary answer
        answer = (
            "Your FAISS + Hugging Face answer "
            "will appear here."
        )

        with st.chat_message("assistant"):
            st.write(answer)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })
import streamlit as st

st.title("VM0042 Agent")

st.write("Hi! Welcome to the VM0042 Question Answering System.")


import faiss

# 1. Load the FAISS index
index = faiss.read_index("index.faiss")

# 2. View key index properties
print(f"Total vectors indexed: {index.ntotal}")
print(f"Vector dimension: {index.d}")
print(f"Is index trained? {index.is_trained}")

# 3. Reconstruct specific vectors (if the index type supports reconstruction)
try:
    # Retrieve the first vector in the index (index 0)
    first_vector = index.reconstruct(0)
    print("Sample vector array:", first_vector[:5])  # Prints first 5 dimensions
except Exception as e:
    print("This index type does not support direct vector reconstruction:", e)
