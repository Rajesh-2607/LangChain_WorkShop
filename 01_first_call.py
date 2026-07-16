# 01_first_call.py
from config import load_dotenv  # noqa (or copy the 3 lines below directly)
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI

# Do not set temperature, top_p, or top_k for Gemini 3.5 per official migration guide.
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

response = llm.invoke("Explain what LangChain is in exactly two sentences.")
print(response.content)
print(response.usage_metadata)  # {'input_tokens':..., 'output_tokens':..., 'total_tokens':...}