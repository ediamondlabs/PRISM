import PyPDF2

pdf_path = "/Users/georgelevis/Projects/DRL v2v anomaly detection/docs/Research_Directions_Document___George_Levis.pdf"
output_path = "/Users/georgelevis/Projects/DRL v2v anomaly detection/pdf_content.txt"

try:
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
    
    with open(output_path, 'w') as f:
        f.write(text)
    print(f"Successfully wrote {len(text)} characters to {output_path}")
except Exception as e:
    print(f"Error reading PDF: {e}")
