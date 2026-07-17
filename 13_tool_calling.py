# 13_tool_calling.py
from config import load_dotenv  # noqa (or copy the 3 lines below directly)

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

@tool
def get_weather(location: str) -> str:
    """Get the current weather for a given location."""
    fake_db = {"boston": "18C, cloudy", "tokyo": "26C, sunny"}
    return fake_db.get(location.lower(), "No data for that location.")

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash")

# bind_tools() tells the model which functions it MAY call
model_with_tools = llm.bind_tools([get_weather])

# Step 1: model decides whether to call a tool
messages = [HumanMessage("What's the weather in Boston?")]
ai_msg = model_with_tools.invoke(messages)
messages.append(ai_msg)
print(ai_msg.tool_calls)  # [{'name': 'get_weather', 'args': {'location': 'Boston'}, ...}]

# Step 2: execute the requested tool(s) and collect results
for tool_call in ai_msg.tool_calls:
    tool_result = get_weather.invoke(tool_call)  # returns a ToolMessage
    messages.append(tool_result)

# Step 3: pass tool results back so the model can produce a final answer
final_response = model_with_tools.invoke(messages)
print(final_response.content)