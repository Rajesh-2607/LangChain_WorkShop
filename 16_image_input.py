# 16_image_input.py
import base64
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash")

# --- Option A: remote image URL ---
message_url = HumanMessage(content=[
    {"type": "text", "text": "Describe this image in one sentence."},
    {"type": "image_url", "image_url": "https://picsum.photos/seed/langchain/300/200"},
])
print(llm.invoke([message_url]).content)

# --- Option B: local image file (base64-encoded) ---
with open("my_photo.jpg", "rb") as f:
    encoded = base64.b64encode(f.read()).decode("utf-8")

message_local = HumanMessage(content=[
    {"type": "text", "text": "What objects do you see in this photo?"},
    {"type": "image_url", "image_url": f"data:image/jpeg;base64,{encoded}"},
])
print(llm.invoke([message_local]).content)