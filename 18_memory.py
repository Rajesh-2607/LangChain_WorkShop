# 18_memory.py
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Remember context across turns."),
    ("placeholder", "{history}"),
    ("human", "{input}"),
])
chain = prompt | llm

# Simple in-memory store keyed by session_id -- swap for Redis/Postgres in prod
store: dict[str, InMemoryChatMessageHistory] = {}

def get_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

chain_with_memory = RunnableWithMessageHistory(
    chain, get_history,
    input_messages_key="input",
    history_messages_key="history",
)

config = {"configurable": {"session_id": "user-123"}}
print(chain_with_memory.invoke({"input": "My name is Sam."}, config=config).content)
print(chain_with_memory.invoke({"input": "What's my name?"}, config=config).content)