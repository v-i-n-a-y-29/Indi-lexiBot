"""
rag_chain.py — Wires the retriever and LLM into a full RAG pipeline.
"""

import sys
import os
import time

# Allow imports from the src/ directory when run from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from retriever import retrieve
from llm import chain


def build_context(docs: list) -> str:
    """Format retrieved documents into a single context string for the LLM."""
    parts = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        header = f"[Source {i}]"
        if meta.get("case_name"):
            header += f" Case: {meta['case_name']}"
        if meta.get("judgment_date"):
            header += f" | Date: {meta['judgment_date']}"
        if meta.get("file_name"):
            header += f" | File: {meta['file_name']}"
        if meta.get("page_number"):
            header += f" | Page: {meta['page_number']}"
        parts.append(f"{header}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def ask(question: str, stream: bool = True) -> str:
    """
    Full RAG pipeline:
      1. Retrieve relevant chunks from FAISS
      2. Build a context string from those chunks
      3. Pass context + question to the LLM and return the answer
    """
    docs = retrieve(question)
    context = build_context(docs)

    if stream:
        print("\nAssistant:\n")
        response_parts = []
        for chunk in chain.stream({"question": question, "context": context}):
            response_parts.append(chunk)
            for char in chunk:
                print(char, end="", flush=True)
                time.sleep(0.02)  # Adjust this value: lower = faster, higher = slower
        print()
        return "".join(response_parts)
    else:
        return chain.invoke({"question": question, "context": context})
