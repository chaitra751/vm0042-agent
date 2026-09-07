import streamlit as st
from pathlib import Path
from pypdf import PdfReader

st.title("VM0042 Agent")
st.write("Hi! Welcome to the VM0042 Question Answering System.")

# Defines the path to the 'data' folder relative to this script file
PDF_FOLDER = Path(__file__).parent / "data"

# Automatically creates the 'data' folder if it doesn't exist yet
PDF_FOLDER.mkdir(exist_ok=True)

# Finds all PDF files in the 'data' folder
pdf_files = sorted(PDF_FOLDER.glob("*.pdf"))

# Sidebar layout
st.sidebar.header("📚 VM0042 Documents")

if not pdf_files:
    st.sidebar.error("No PDF files found in the 'data' folder.")
    st.info("Please place your VM0042 PDF documents into the `data` directory.")
else:
    selected_pdf = st.sidebar.selectbox(
        "Select PDF",
        pdf_files,
        format_func=lambda x: x.name
    )

    st.header(f"📄 {selected_pdf.name}")

    reader = PdfReader(selected_pdf)
    document_text = ""

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text()
        if page_text:
            document_text += f"\n\n--- Page {page_number} ---\n\n" + page_text

    st.text_area(
        "PDF Content",
        document_text,
        height=650
    )
