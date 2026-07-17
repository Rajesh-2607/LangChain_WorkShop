# 17_video_input.py
from google import genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# 1. Upload file using the underlying SDK
client = genai.Client()
myfile = client.files.upload(file="my_video.mp4")

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash")

# 2. Reference the uploaded file by its URI in a HumanMessage
message = HumanMessage(content=[
    {"type": "text", "text": "Summarize this video in 3 sentences."},
    {
        "type": "file",
        "file_id": myfile.uri,
        "mime_type": "video/mp4",
    },
])

response = llm.invoke([message])
print(response.text)