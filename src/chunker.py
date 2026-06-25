import json
import hashlib
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

INPUT_FILE  = Path("processed/clean_documents.jsonl")
OUTPUT_FILE = Path("processed/chunks.jsonl")

# Chunking parameters — tune these for legal text
CHUNK_SIZE    = 800   # characters per chunk
CHUNK_OVERLAP = 150   # overlap to preserve context across boundaries

def load_documents(path: Path) -> list[Document]:
    """Load clean_documents.jsonl and return a list of LangChain Documents."""
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            doc = Document(
                page_content=record["text"],
                metadata={
                    **record["metadata"],
                    "doc_id": record["doc_id"],  # carry the original page-level id
                },
            )
            docs.append(doc)
    return docs


def make_chunk_id(doc_id: str, chunk_index: int) -> str:
    """Produce a stable, unique chunk id from the parent doc_id and its index."""
    raw = f"{doc_id}::chunk_{chunk_index}"
    return hashlib.md5(raw.encode()).hexdigest()


def chunk_documents(docs: list[Document]) -> list[dict]:
    """Split documents and return a list of serialisable chunk records."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # These separators respect legal paragraph / sentence structure
        separators=["\n\n", "\n", ". ", ", ", " ", ""],
        length_function=len,
    )

    output_records = []

    for doc in docs:
        chunks = splitter.split_documents([doc])

        for idx, chunk in enumerate(chunks):
            doc_id      = chunk.metadata.get("doc_id", "unknown")
            chunk_index = idx
            chunk_id    = make_chunk_id(doc_id, chunk_index)
            chunk_text  = chunk.page_content

            record = {
                "chunk_id":    chunk_id,
                "doc_id":      doc_id,
                "chunk_index": chunk_index,
                "chunk_length": len(chunk_text),
                "text":        chunk_text,
                "metadata":    chunk.metadata,
            }
            output_records.append(record)

    return output_records


def save_chunks(records: list[dict], path: Path) -> None:
    """Write chunk records to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_chunker():
    print(f"Loading documents from {INPUT_FILE} …")
    docs = load_documents(INPUT_FILE)
    print(f"Loaded {len(docs)} page-level documents.")

    print("Chunking …")
    chunks = chunk_documents(docs)
    print(f"Produced {len(chunks)} chunks.")

    print(f"Saving chunks to {OUTPUT_FILE} …")
    save_chunks(chunks, OUTPUT_FILE)
    print("Done.")

    # Quick summary
    domains = {}
    for c in chunks:
        d = c["metadata"].get("domain", "unknown")
        domains[d] = domains.get(d, 0) + 1
    print("\nChunks per domain:")
    for domain, count in sorted(domains.items()):
        print(f"  {domain}: {count}")


if __name__ == "__main__":
    run_chunker()
