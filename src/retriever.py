# pyrefly: ignore [missing-import]
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# 1. Load the exact same embedding model used to create the index
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

# 2. Load the saved FAISS index
vectorstore = FAISS.load_local(
    "vectorstore/faiss_index", 
    embeddings, 
    allow_dangerous_deserialization=True
)

# 3. Perform search
query = "what is the penalty for drink and drive"

# Using Maximal Marginal Relevance (MMR) search prevents fetching duplicate/overlapping chunks
# It fetches 10 documents, then selects the top 3 that are most diverse from each other
results = vectorstore.max_marginal_relevance_search(query, k=3, fetch_k=10)

# 4. Display the results
print(f"Query: {query}\n")
print("Relevant Documents (Diverse):")

for i, doc in enumerate(results):
    print(f"\nDocument {i + 1}")
    print(f"Case: {doc.metadata.get('case_name', 'Unknown')}")
    print("-" * 50)
    print(doc.page_content)