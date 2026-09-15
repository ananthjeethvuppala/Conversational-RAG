import os
from pypdf import PdfReader

def load_pdfs(foulder_path):
    documents = []

    for file_name in os.listdir(foulder_path):

        if file_name.endswith(".pdf"):
            file_path = os.path.join(foulder_path, file_name)

        reader = PdfReader(file_path)
        text = ""

        for page in reader.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text
                text += "/n"

        documents.append(
            {
                "text" : text,
                "source" : file_name 
            }
        )
    return documents