"""
Stage 3 — Embedding & FAISS Index
==================================
Loads processed/chunks.jsonl → generates embeddings → saves FAISS index.

To swap in InLegal-SBERT (once available) set the environment variable:
    EMBEDDING_MODEL=law-ai/InLegalBERT
or edit EMBEDDING_MODEL_NAME below.
"""

import json
import os
import time
from pathlib import Path

from langchain_core.documents import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS  

# ---------------------------------------------------------------------------
# Configuration — change EMBEDDING_MODEL_NAME to swap models at any time.
# When InLegal-SBERT becomes available, set it to "law-ai/InLegalBERT" or
# the fine-tuned SBERT checkpoint path.
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL)

CHUNKS_FILE    = Path("processed/chunks.jsonl")
FAISS_DIR      = Path("vectorstore/faiss_index")
REPORT_FILE    = Path("processed/embedding_report.json")

# Batch size for embedding calls — reduce if you hit memory limits
BATCH_SIZE = 64


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_chunks(path: Path) -> list[dict]:
    """Read all records from chunks.jsonl."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_documents(records: list[dict]) -> list[Document]:
    """
    Convert chunk records into LangChain Documents.
    Metadata is fully populated so every chunk is self-contained for retrieval.
    """
    docs = []
    for r in records:
        base_meta = r.get("metadata", {})
        meta = {
            # Chunk-level identifiers
            "chunk_id":     r.get("chunk_id", ""),
            "doc_id":       r.get("doc_id", ""),
            "chunk_index":  r.get("chunk_index", 0),
            "chunk_length": r.get("chunk_length", 0),
            # Document-level provenance (from upstream metadata)
            "source_path":   base_meta.get("source_path", ""),
            "file_name":     base_meta.get("file_name", ""),
            "domain":        base_meta.get("domain", ""),
            "case_name":     base_meta.get("case_name", ""),
            "judgment_date": base_meta.get("judgment_date", ""),
            "page_number":   base_meta.get("page_number", None),
        }
        docs.append(Document(page_content=r["text"], metadata=meta))
    return docs


def get_embedding_dimension(model: HuggingFaceEmbeddings) -> int:
    """Embed a dummy string to detect the output dimension."""
    sample = model.embed_query("test")
    return len(sample)


def save_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_embedding_pipeline():
    report: dict = {
        "total_chunks_loaded":   0,
        "total_chunks_embedded": 0,
        "embedding_model":       EMBEDDING_MODEL_NAME,
        "embedding_dimension":   None,
        "faiss_index_path":      str(FAISS_DIR),
        "failed_chunks":         [],
        "duration_seconds":      None,
    }

    start = time.time()

    # 1. Load chunks --------------------------------------------------------
    print(f"Loading chunks from {CHUNKS_FILE} …")
    records = load_chunks(CHUNKS_FILE)
    report["total_chunks_loaded"] = len(records)
    print(f"  Loaded {len(records)} chunks.")

    # 2. Build LangChain Documents ------------------------------------------
    print("Building LangChain Documents …")
    docs = build_documents(records)

    # 3. Initialise the embedding model -------------------------------------
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    print("  (first run will download weights — may take a minute)")
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"batch_size": BATCH_SIZE, "normalize_embeddings": True},
    )

    dim = get_embedding_dimension(embedding_model)
    report["embedding_dimension"] = dim
    print(f"  Embedding dimension: {dim}")

    # 4. Build FAISS vector store -------------------------------------------
    print(f"Generating embeddings and building FAISS index …")
    print(f"  Processing {len(docs)} chunks in batches of {BATCH_SIZE} …")

    failed: list[str] = []

    # Process in batches to give progress feedback and handle errors cleanly
    vectorstore = None
    for batch_start in range(0, len(docs), BATCH_SIZE):
        batch = docs[batch_start : batch_start + BATCH_SIZE]
        try:
            if vectorstore is None:
                vectorstore = FAISS.from_documents(batch, embedding_model)
            else:
                vectorstore.add_documents(batch)

            done = min(batch_start + BATCH_SIZE, len(docs))
            print(f"  Embedded {done}/{len(docs)} chunks …", end="\r")

        except Exception as e:
            for doc in batch:
                cid = doc.metadata.get("chunk_id", "unknown")
                failed.append({"chunk_id": cid, "error": str(e)})
            print(f"\n  Warning: batch starting at {batch_start} failed — {e}")

    print()  # newline after \r progress

    report["failed_chunks"]         = failed
    report["total_chunks_embedded"] = len(docs) - len(failed)

    # 5. Save FAISS index ---------------------------------------------------
    if vectorstore:
        FAISS_DIR.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(FAISS_DIR))
        print(f"FAISS index saved to {FAISS_DIR}/")
    else:
        print("No vectorstore created — all batches failed.")

    # 6. Save report --------------------------------------------------------
    report["duration_seconds"] = round(time.time() - start, 2)
    save_report(report, REPORT_FILE)
    print(f"Embedding report saved to {REPORT_FILE}")

    # Summary ---------------------------------------------------------------
    print("\n=== Embedding Pipeline Complete ===")
    print(f"  Chunks loaded:    {report['total_chunks_loaded']}")
    print(f"  Chunks embedded:  {report['total_chunks_embedded']}")
    print(f"  Failed chunks:    {len(failed)}")
    print(f"  Model:            {report['embedding_model']}")
    print(f"  Dimension:        {report['embedding_dimension']}")
    print(f"  Duration:         {report['duration_seconds']}s")
    print(f"  FAISS index:      {FAISS_DIR}/")


if __name__ == "__main__":
    run_embedding_pipeline()
