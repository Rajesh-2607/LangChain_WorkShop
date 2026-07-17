# 14_google_search_grounding.py
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash")

# Bind Gemini's NATIVE Google Search tool -- no external API key needed,
# it's built into the Gemini API itself and grounds answers in live results.
# Note: Using bind_tools for google_search like this is unconfirmed in docs -- verify before production use.
# The officially documented approach is often .bind(tools=[{"google_search": {}}], ...)
model_with_search = llm.bind_tools([{"google_search": {}}])

response = model_with_search.invoke(
    "When is the next total solar eclipse visible from the US?"
)
print(response.content)
# response.content_blocks also exposes grounding metadata/citations