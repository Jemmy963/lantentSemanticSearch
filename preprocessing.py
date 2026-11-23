from pypdf import PdfReader

def extract_pdf_text (pdf_file):
    #Extract text from all pages of the uploaded pdf.
    reader = PdfReader(pdf_file)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text() or "" 
        text += page_text + " " 
    

    return text