# 07_few_shot.py
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash")

# Provide labeled examples so the model mimics the desired output style/format
examples = [
    {"input": "I love this product!", "output": "positive"},
    {"input": "This is terrible, waste of money.", "output": "negative"},
    {"input": "It's okay, does the job.", "output": "neutral"},
]

example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}"),
])

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

final_prompt = ChatPromptTemplate.from_messages([
    ("system", "Classify sentiment as exactly one word: positive, negative, or neutral."),
    few_shot_prompt,
    ("human", "{input}"),
])

chain = final_prompt | llm
print(chain.invoke({"input": "Shipping was fast but the box arrived damaged."}).content)