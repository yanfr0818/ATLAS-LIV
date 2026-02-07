#!/usr/bin/env python
"""
Extract text from ATL-DAPR-PUB-2021-001.pdf
"""
import fitz

pdf_path = r"D:\HEP\ATLAS\LIV\ATL-DAPR-PUB-2021-001.pdf"
doc = fitz.open(pdf_path)

print(f"Total pages: {len(doc)}")

# Extract first 10 pages
for i in range(min(10, len(doc))):
    text = doc[i].get_text()
    print(f"\n{'='*60}")
    print(f"PAGE {i+1}")
    print("="*60)
    print(text[:4000] if len(text) > 4000 else text)
