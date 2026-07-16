# 11_structured_output.py
from pydantic import BaseModel, Field
from typing import Literal
from langchain_google_genai import ChatGoogleGenerativeAI

class Feedback(BaseModel):
    """Schema the model MUST conform to."""
    sentiment: Literal["positive", "neutral", "negative"] = Field(
        description="Overall sentiment of the feedback"
    )
    summary: str = Field(description="One-sentence summary of the feedback")

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash")

# with_structured_output constrains generation to match the Pydantic schema.
# method="json_schema" uses Gemini's native response_json_schema API param
# (recommended -- constrains generation directly, more reliable).
structured_llm = llm.with_structured_output(Feedback, method="json_schema")

result = structured_llm.invoke("The new UI is great, super intuitive!")
print(type(result))       # 
print(result.sentiment)   # positive
print(result.summary)