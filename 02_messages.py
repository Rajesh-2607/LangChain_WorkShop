# 02_messages.py
from config import load_dotenv  # noqa (or copy the 3 lines below directly)
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash")

messages = [
    SystemMessage(content="You are a concise, friendly Python tutor."),
    HumanMessage(content="What is a list comprehension?"),
]

response = llm.invoke(messages)
print("AI:", response.content)

# Multi-turn: append the AI reply, then ask a follow-up in the same context
messages.append(AIMessage(content=response.content))
messages.append(HumanMessage(content="Show one example that also filters values."))
print("AI:", llm.invoke(messages).content)