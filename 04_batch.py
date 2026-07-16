# 04_batch.py
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash")

# .batch() sends multiple independent prompts concurrently — much faster
# than looping .invoke() calls one at a time.
prompts = [
    "Give a one-line definition of 'API'.",
    "Give a one-line definition of 'SDK'.",
    "Give a one-line definition of 'LLM'.",
]

results = llm.batch(prompts)
for p, r in zip(prompts, results):
    print(f"{p} -> {r.content}")