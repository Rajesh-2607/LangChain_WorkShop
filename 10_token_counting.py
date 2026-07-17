# 10_token_counting.py
from config import load_dotenv  # noqa (or copy the 3 lines below directly)

from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash")

text = "How many tokens does this sentence use on the Gemini tokenizer?"

# get_num_tokens uses the model's own tokenizer -- useful for checking
# whether an input will fit inside the model's context window before sending it.
print("Estimated input tokens:", llm.get_num_tokens(text))

response = llm.invoke(text)
# usage_metadata gives the ACTUAL counts returned by the API after the call
print("Actual usage:", response.usage_metadata)