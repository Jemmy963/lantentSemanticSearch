import streamlit as st

from preprocessing import extract_pdf_text
from matrix import clean_text, create_tfidf_matrix
from svdModel import apply_svd
from search import search_documents

st.title("Latent Semantic Search Engine")
st.write("Upload PDF documents and search through their contents.")

uploaded_files = st.file_uploader(
    "Upload your PDF documents",
    type=["pdf"],
    accept_multiple_files=True,
)

if uploaded_files:
    filenames = []
    documents = []

    for pdf in uploaded_files:
        extracted_text = extract_pdf_text(pdf)
        cleaned_text = clean_text(extracted_text)

        if cleaned_text.strip():
            filenames.append(pdf.name)
            documents.append(cleaned_text)

    if len(documents) < 2:
        st.warning("Please upload at least two PDFs containing readable text.")
    else:
        try:
            vectoriser, tfidf_matrix = create_tfidf_matrix(documents)
            svd, document_vectors = apply_svd(tfidf_matrix)
        except ValueError as error:
            if "different terms" in str(error):
                st.warning(
                    "The PDFs need at least two different terms after text cleaning. "
                    "Try uploading documents with more varied text."
                )
            else:
                raise
        else:
            st.success("Documents have been processed successfully!")
            query = st.text_input("Enter your search query")

            if st.button("Search"):
                if query.strip():
                    results = search_documents(
                        query,
                        vectoriser,
                        svd,
                        document_vectors,
                        filenames,
                    )
                    st.subheader("Search Results")
                    for filename, score in results:
                        st.write(f"**Document:** {filename}")
                        st.write(f"Similarity score: {score:.4f}")
                        st.divider()
                else:
                    st.warning("Please enter a search query.")
