from pypdf import PdfReader

pdf_path = "../data/computer_fundamental.pdf"

reader = PdfReader(pdf_path)

all_text = []

for page_number, page in enumerate(reader.pages, start=1):
    text = page.extract_text()
    if text:
        all_text.append(f"\n--- Page {page_number} ---\n{text}")

full_text = "\n".join(all_text)

output_path = "../data/computer_fundamental.txt"
with open(output_path, "w", encoding="utf-8") as f:
    f.write(full_text)

print(f"Extracted text saved to {output_path}")
print(f"Total pages processed: {len(reader.pages)}")
print(f"Total characters extracted: {len(full_text)}")