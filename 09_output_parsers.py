# 09_output_parsers.py
from config import load_dotenv  # noqa (or copy the 3 lines below directly)

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash")

# JsonOutputParser expects the model to return valid JSON text and
# parses it into a Python dict automatically.
parser = JsonOutputParser()

prompt = ChatPromptTemplate.from_messages([
    ("system", "Return ONLY valid JSON with keys 'name' and 'age'. No extra text."),
    ("human", "{text}"),
])

chain = prompt | llm | parser
result = chain.invoke({"text": "John is 34 years old."})
print(result, type(result))  # {'name': 'John', 'age': 34} 