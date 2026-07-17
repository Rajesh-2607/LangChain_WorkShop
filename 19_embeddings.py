# 19_embeddings.py
from config import load_dotenv  # noqa (or copy the 3 lines below directly)

from langchain_google_genai import GoogleGenerativeAIEmbeddings

# gemini-embedding-001 is the current stable, text-only embedding model
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

vector = embeddings.embed_query("hello, world!")
print(len(vector), vector[:5])

# Batch-embed multiple documents at once (max ~100 strings per call)
docs = ["Today is Monday", "Today is Tuesday", "Today is April Fools day"]
doc_vectors = embeddings.embed_documents(docs)
print(len(doc_vectors), len(doc_vectors[0]))

# task_type tunes the embedding for a specific use case (retrieval, classification, etc.)
query_embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", task_type="RETRIEVAL_QUERY")
doc_embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", task_type="RETRIEVAL_DOCUMENT")

from sklearn.metrics.pairwise import cosine_similarity
q = query_embeddings.embed_query("What is the capital of France?")
d = doc_embeddings.embed_documents(["The capital of France is Paris.", "Cats are great pets."])
for i, vec in enumerate(d):
    print(f"doc {i+1} similarity:", cosine_similarity([q], [vec])[0][0])