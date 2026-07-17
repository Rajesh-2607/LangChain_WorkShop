# capstone3_research_agent.py
# ==============================================================
# CAPSTONE 3: A Gemini-powered research agent with:
#   - native Google Search grounding tool (real-time web results)
#   - a calculator tool
#   - persistent memory across turns via LangGraph's MemorySaver
# ==============================================================
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression, e.g. '12 * (4 + 3)'."""
    try:
        # Restricted eval -- for production use a proper parser like `numexpr`
        return str(eval(expression, {"__builtins__": {}}))
    except Exception as e:
        return f"Error evaluating expression: {e}"

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash")

# Bind Gemini's native Google Search tool ALONGSIDE our custom calculator tool
# -- the model decides which (if any) to call at each step.
tools = [calculator, {"google_search": {}}]

# MemorySaver persists conversation state per thread_id, so the agent
# remembers earlier turns in the same session.
checkpointer = MemorySaver()
agent = create_react_agent(llm, tools, checkpointer=checkpointer)

def ask_agent(question: str, thread_id: str = "session-1") -> str:
    """Run one question through the agent, preserving memory across calls."""
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke({"messages": [("human", question)]}, config=config)
    return result["messages"][-1].content

if __name__ == "__main__":
    print("Research Agent ready. Type 'exit' to quit.\n")
    while True:
        q = input("You: ")
        if q.lower() in {"exit", "quit"}:
            break
        print("Agent:", ask_agent(q), "\n")

    # Try:
    # - "What year did the current Wimbledon champion first turn pro? Multiply that year by 2."
    #     -> agent should search the web, extract a number, then call the calculator.
    # - Ask a follow-up like "and who did they beat in the final?" -- memory persists
    #   context from the previous turn thanks to the checkpointer + thread_id.