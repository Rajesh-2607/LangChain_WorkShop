# capstone1_pdf_rag.py
# ==============================================================
# CAPSTONE 1: Chat with a PDF using LangChain + Gemini (RAG)
# Pipeline: load -> split -> embed -> retrieve -> generate
# ==============================================================
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

PDF_PATH = "sample.pdf"  # put any PDF next to this script

# --- 1. Load: one Document object per page ---
raw_docs = PyPDFLoader(PDF_PATH).load()

# --- 2. Split into overlapping chunks for better retrieval granularity ---
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = splitter.split_documents(raw_docs)
print(f"Loaded {len(raw_docs)} pages -> {len(chunks)} chunks")

# --- 3 & 4. Embed + index ---
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
vector_store = InMemoryVectorStore.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 4})

# --- 5. Prompt that forces grounded answers ---
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Answer using ONLY the provided context. If the answer isn't in the "
     "context, say you don't know. Cite page numbers when available.\n\n"
     "Context:\n{context}"),
    ("human", "{question}"),
])

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash")

def format_docs(docs):
    return "\n\n".join(f"[page {d.metadata.get('page', '?')}] {d.page_content}" for d in docs)

# --- 6. Full RAG chain via LCEL ---
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)

if __name__ == "__main__":
    print("PDF RAG Chatbot ready. Type 'exit' to quit.\n")
    while True:
        question = input("You: ")
        if question.lower() in {"exit", "quit"}:
            break
        print("Bot:", rag_chain.invoke(question), "\n") 