# 06_prompt_templates.py
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash")

# {placeholders} get filled at call time via .invoke({...})
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert copywriter who writes in a {tone} tone."),
    ("human", "Write a 2-line product tagline for: {product}"),
])

chain = prompt | llm  # pipe the prompt straight into the model (LCEL)

result = chain.invoke({"tone": "playful", "product": "a smart coffee mug"})
print(result.content)