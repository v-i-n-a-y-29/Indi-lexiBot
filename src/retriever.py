# pyrefly: ignore [missing-import]
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Project root is one level up from this file (src/ -> project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FAISS_INDEX_PATH = str(PROJECT_ROOT / "vectorstore" / "faiss_index")

# Load the same embedding model used to build the index
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

# Load the saved FAISS index once at import time
vectorstore = FAISS.load_local(
    FAISS_INDEX_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)


def retrieve(query: str, k: int = 3, fetch_k: int = 10) -> list:
    """
    Return the top-k most relevant and diverse documents for a query.
    Uses MMR (Maximal Marginal Relevance) to avoid redundant chunks.
    """
    return vectorstore.max_marginal_relevance_search(query, k=k, fetch_k=fetch_k)