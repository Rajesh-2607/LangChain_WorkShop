# 15_code_execution.py
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash")

# Gemini can write AND run Python internally via the code_execution tool --
# great for precise math/data tasks instead of relying on the LLM's own arithmetic.
model_with_code_interpreter = llm.bind_tools([{"code_execution": {}}])

response = model_with_code_interpreter.invoke("Use Python to calculate 17 factorial.")
print(response.content_blocks)  # returns {'type': 'server_tool_call', 'name': 'code_interpreter', ...} and {'type': 'server_tool_result', ...}