"""
document_rag.py
----------------
Support code for RAG (Retrieval-Augmented Generation) question answering.

RAG in three steps:
  1. EXTRACT + CHUNK : read a PDF/DOCX/TXT file and cut the text into small chunks.
  2. RETRIEVE        : for a question, find the few chunks most similar to it.
  3. GENERATE        : give only those chunks to the LLM (done in llm_service.py).

Retrieval uses TF-IDF + cosine similarity from scikit-learn: simple, fast, needs
no model download, and is easy to explain. (A later upgrade could use embeddings.)
"""

import io

from docx import Document
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SUPPORTED_DOCUMENT_TYPES = [".pdf", ".docx", ".txt"]


def extract_text(file_name, file_bytes):
    """Read the plain text out of a PDF, DOCX or TXT file."""
    extension = "." + file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""

    if extension == ".pdf":
        reader = PdfReader(io.BytesIO(file_bytes))
        text = "\n".join((page.extract_text() or "") for page in reader.pages)
    elif extension == ".docx":
        document = Document(io.BytesIO(file_bytes))
        text = "\n".join(paragraph.text for paragraph in document.paragraphs)
    elif extension == ".txt":
        text = file_bytes.decode("utf-8", errors="ignore")
    else:
        raise ValueError("Unsupported document type. Please upload a PDF, DOCX or TXT file.")

    if not text.strip():
        raise ValueError("No readable text found in this document (a scanned PDF has no text layer).")
    return text


def split_into_chunks(text, words_per_chunk=200, overlap_words=40):
    """
    Cut text into chunks of about 200 words. Consecutive chunks overlap by 40 words
    so a sentence that falls on a boundary is not lost.
    """
    words = text.split()
    chunks = []
    step = words_per_chunk - overlap_words
    for start in range(0, len(words), step):
        chunk = " ".join(words[start:start + words_per_chunk])
        if chunk:
            chunks.append(chunk)
        if start + words_per_chunk >= len(words):
            break
    return chunks


def find_relevant_chunks(question, chunks, top_k=4):
    """
    Return the top_k chunks most similar to the question (best first).
    Chunks with zero similarity (no shared words) are dropped.
    """
    if not chunks:
        return []
    try:
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        chunk_vectors = vectorizer.fit_transform(chunks)      # one vector per chunk
        question_vector = vectorizer.transform([question])    # same vocabulary for the question
    except ValueError:
        return []  # happens when the text only contains stop words
    scores = cosine_similarity(question_vector, chunk_vectors)[0]
    best_first = scores.argsort()[::-1][:top_k]
    return [chunks[i] for i in best_first if scores[i] > 0]
