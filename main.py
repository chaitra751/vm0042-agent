import streamlit as st

st.title("VM0042 Agent")

st.write("Hi! Welcome to the VM0042 Question Answering System.")


PDF_FOLDER = Path("data")

pdf_files = sorted(PDF_FOLDER.glob("*.pdf"))

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("📚 VM0042 Documents")

if not pdf_files:
    st.sidebar.error("No PDF files found in data folder.")
else:

    selected_pdf = st.sidebar.selectbox(
        "Select PDF",
        pdf_files,
        format_func=lambda x: x.name
    )

    st.sidebar.success(f"{len(pdf_files)} PDFs available")

    # -----------------------------
    # Display selected PDF
    # -----------------------------
    st.header(f"📄 {selected_pdf.name}")

    try:
        reader = PdfReader(selected_pdf)

        document_text = ""

        for page_number, page in enumerate(reader.pages, start=1):

            page_text = page.extract_text()

            if page_text:
                document_text += (
                    f"\n\n--- Page {page_number} ---\n\n"
                    + page_text
                )

        st.text_area(
            "PDF Content",
            document_text,
            height=650
        )

    except Exception as e:
        st.error(f"Error reading PDF: {e}")
