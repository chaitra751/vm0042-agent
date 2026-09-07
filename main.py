import streamlit as st
from pathlib import Path
from pypdf import PdfReader

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
st.title("🌱 VM0042 Agent")
st.write("Hi! Welcome to the VM0042 Question Answering System.")

# ==========================================
# DATA FOLDER PATH
# ==========================================
PDF_FOLDER = Path(__file__).parent / "data"

# ==========================================
# FIND ALL PDF FILES
# ==========================================
pdf_files = sorted(PDF_FOLDER.glob("*.pdf"))

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.title("📚 VM0042 Documents")

if not pdf_files:

    st.sidebar.error("No PDF files found.")

    st.error(
        f"No PDF files were found in:\n\n{PDF_FOLDER}"
    )

else:

    st.sidebar.success(
        f"{len(pdf_files)} PDF files found"
    )

    # PDF dropdown
    selected_pdf = st.sidebar.selectbox(
        "Select a PDF",
        pdf_files,
        format_func=lambda x: x.name
    )

    # ==========================================
    # MAIN CONTENT
    # ==========================================
    st.header("📄 Selected Document")

    st.write(f"**{selected_pdf.name}**")

    # ==========================================
    # READ PDF
    # ==========================================
    try:

        reader = PdfReader(str(selected_pdf))

        total_pages = len(reader.pages)

        st.info(
            f"Total Pages: {total_pages}"
        )

        document_text = ""

        # Extract text from every page
        for page_number, page in enumerate(
            reader.pages,
            start=1
        ):

            page_text = page.extract_text()

            if page_text:

                document_text += (
                    f"\n\n"
                    f"==============================\n"
                    f"PAGE {page_number}\n"
                    f"==============================\n\n"
                    f"{page_text}"
                )

        # ==========================================
        # DISPLAY PDF CONTENT
        # ==========================================
        st.subheader("📖 Document Content")

        if document_text.strip():

            st.text_area(
                "Extracted PDF Text",
                document_text,
                height=650
            )

        else:

            st.warning(
                "No text could be extracted from this PDF. "
                "It may be a scanned/image-based PDF."
            )

    except Exception as e:

        st.error(
            f"Error reading PDF: {str(e)}"
        )
