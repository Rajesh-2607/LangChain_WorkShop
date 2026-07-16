# 08_lcel_chain.py
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash")

prompt = ChatPromptTemplate.from_messages([
    ("system", "Summarize the user's text in exactly 3 bullet points."),
    ("human", "{text}"),
])

# StrOutputParser extracts the plain string from the AIMessage
parser = StrOutputParser()

# The "|" pipe composes prompt -> model -> parser into ONE Runnable chain
chain = prompt | llm | parser

long_text = ("LangChain is a framework that simplifies building applications "
             "powered by large language models, providing abstractions for "
             "prompts, chains, memory, agents, and vector store integrations.")

print(chain.invoke({"text": long_text}))