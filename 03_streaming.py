# 03_streaming.py
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash")

print("Gemini: ", end="", flush=True)
for chunk in llm.stream("Write a haiku about debugging code."):
    # Each chunk is an AIMessageChunk; .content holds the partial text
    print(chunk.content, end="", flush=True)
print()