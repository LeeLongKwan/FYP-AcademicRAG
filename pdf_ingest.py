#!/usr/bin/env python3
"""
PDF Ingestion - STRICT "."-BASED CHUNKING + ADVANCED DIAGRAM & TABLE DETECTION
- Chunks ONLY when a "." is detected (strict sentence boundary)
- Never breaks a sentence
- Detects images, diagrams, figures, and tables
- Adds clear placeholders for visual content
- Preserves [SOURCE: ...] headers
"""
import os
import glob
import re
import argparse
from tqdm import tqdm
import fitz  # PyMuPDF

# CONFIG
DEFAULT_PDF_DIR = "./pdfs"
DEFAULT_CHUNK_SIZE = 850
DEFAULT_OVERLAP = 180
OUTPUT_FILE = "chunks.txt"


def clean_text(text: str) -> str:
    """Clean whitespace while preserving structure."""
    text = re.sub(r'\s+', ' ', text.strip())
    return text


def split_by_period(text: str):
    """Strict sentence splitter: chunks ONLY when '.' is detected."""
    sentences = re.split(r'(?<=\.)\s+', text)
    return [s.strip() for s in sentences if s.strip() and s.strip() != '.']


def chunk_by_period_sentences(text: str, chunk_size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_OVERLAP):
    """Chunk strictly by sentences ending with '.' + overlap."""
    if not text:
        return []

    sentences = split_by_period(text)
    chunks = []
    current_chunk = []
    current_len = 0

    for sent in sentences:
        sent_len = len(sent)
        if current_len + sent_len + 2 > chunk_size and current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(chunk_text.strip())

            # Overlap: keep last few sentences
            overlap_sentences = current_chunk[-3:] if len(current_chunk) > 3 else current_chunk[-2:]
            current_chunk = overlap_sentences
            current_len = sum(len(s) + 2 for s in overlap_sentences)

        current_chunk.append(sent)
        current_len += sent_len + 2

    if current_chunk:
        chunks.append(" ".join(current_chunk).strip())

    # Safety filter
    chunks = [c for c in chunks if len(c) > 80]
    return chunks


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text + detect diagrams, figures, images, and tables."""
    doc = fitz.open(pdf_path)
    all_pages = []

    for page_num in range(len(doc)):
        page = doc[page_num]

        # 1. Get normal text
        page_text = page.get_text("text")
        page_text = clean_text(page_text)

        # 2. Detect visual content
        visual_notes = []

        # Images / Diagrams
        image_count = len(page.get_images())
        if image_count > 0:
            visual_notes.append(f"[DIAGRAM / FIGURE / IMAGE detected on this page - {image_count} image(s) found]")

        # Tables (using PyMuPDF's table detector)
        tables = page.find_tables()
        if tables:
            for i, table in enumerate(tables, 1):
                rows = len(table.extract())
                cols = len(table.extract()[0]) if rows > 0 else 0
                visual_notes.append(f"[TABLE detected on this page - Table {i}: {rows} rows × {cols} columns]")

        # Figure/Table captions (fallback regex)
        if re.search(r'(?i)(figure|fig\.|diagram|plot|graph|table)\s*\d+', page_text):
            visual_notes.append("[FIGURE / TABLE caption detected on this page]")

        header = f"[SOURCE: {os.path.basename(pdf_path)} | PAGE: {page_num + 1}]\n"

        if visual_notes:
            visual_block = "\n".join(visual_notes) + "\n"
            page_content = header + visual_block + page_text
        else:
            page_content = header + page_text

        if page_text or visual_notes:
            all_pages.append(page_content)

    doc.close()
    return "\n\n".join(all_pages)


def main():
    parser = argparse.ArgumentParser(description="PDF ingestion with STRICT '.'-based sentence chunking + diagram & table detection")
    parser.add_argument("--pdf-dir", default=DEFAULT_PDF_DIR)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    args = parser.parse_args()

    pdf_files = sorted(glob.glob(os.path.join(args.pdf_dir, "*.pdf")))
    if not pdf_files:
        print(f"❌ No PDFs found in '{args.pdf_dir}'")
        return

    print(f"📂 Found {len(pdf_files)} PDF(s)")
    print(f"Settings → Chunk size: {args.chunk_size} | Overlap: {args.overlap}\n")

    all_chunks = []
    for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
        pdf_name = os.path.basename(pdf_path)
        try:
            print(f" 📄 Extracting: {pdf_name}")
            full_text = extract_text_from_pdf(pdf_path)
            chunks = chunk_by_period_sentences(full_text, args.chunk_size, args.overlap)
            all_chunks.extend(chunks)
            print(f" → {len(chunks)} '.'-based chunks (with diagram & table detection)")
        except Exception as e:
            print(f" ❌ Error processing {pdf_name}: {e}")

    # Write to chunks.txt
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(chunk.replace("\n", " ").strip() + "\n")

    print("\n" + "="*80)
    print("✅ STRICT '.'-BASED INGESTION WITH DIAGRAM & TABLE DETECTION COMPLETE!")
    print(f" Total chunks created : {len(all_chunks)}")
    print(f" Output file          : {OUTPUT_FILE}")
    print(f" Next step            : python build_index.py")
    print("="*80)


if __name__ == "__main__":
    main()