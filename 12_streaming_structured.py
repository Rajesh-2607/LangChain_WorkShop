# 12_streaming_structured.py
from config import load_dotenv  # noqa (or copy the 3 lines below directly)

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

class Feedback(BaseModel):
    sentiment: str
    summary: str

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash")
structured_model = llm.with_structured_output(
    schema=Feedback.model_json_schema(), method="json_schema"
)

# IMPORTANT: streaming structured output returns partial dicts.
# Merge with .update() instead of "+=" since these are dict fragments,
# not plain strings.
stream = structured_model.stream("The interface is intuitive and beautiful!")
full = next(stream)
for chunk in stream:
    full.update(chunk)

print(full)  # -> {'sentiment': 'positive', 'summary': 'The user praises...'}