# 20_vector_store.py
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

texts = [
    "LangChain helps build LLM-powered applications.",
    "Gemini is Google's multimodal foundation model.",
    "Paris is the capital of France.",
]

# InMemoryVectorStore needs zero extra infra -- perfect for prototyping.
# Swap for FAISS/Chroma/Pinecone when you need persistence at scale.
vector_store = InMemoryVectorStore.from_texts(texts, embedding=embeddings)

retriever = vector_store.as_retriever()
results = retriever.invoke("What is Gemini?")
print(results[0].page_content)