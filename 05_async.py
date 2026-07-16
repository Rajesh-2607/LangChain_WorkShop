# 05_async.py
import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash")

async def main():
    # ainvoke is the async counterpart of invoke — use it inside async
    # web servers (FastAPI, etc.) so you don't block the event loop.
    response = await llm.ainvoke("Name three benefits of async I/O.")
    print(response.content)

    # abatch runs multiple async calls concurrently
    results = await llm.abatch(["Define REST.", "Define gRPC."])
    for r in results:
        print(r.content)

asyncio.run(main())