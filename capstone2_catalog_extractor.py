# capstone2_catalog_extractor.py
# ==============================================================
# CAPSTONE 2: Turn a folder of product images into a structured
# CSV catalog using Gemini's vision + structured output.
# ==============================================================
import os
import csv
import base64
from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

class ProductEntry(BaseModel):
    """Schema for one row of the generated catalog."""
    name: str = Field(description="Short product name")
    category: str = Field(description="e.g. Electronics, Apparel, Furniture")
    description: str = Field(description="1-2 sentence marketing description")
    estimated_price_usd: float = Field(description="Rough estimated retail price in USD")

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash")

# Constrain output to our schema using Gemini's native JSON schema mode
structured_llm = llm.with_structured_output(ProductEntry, method="json_schema")

def encode_image(path: str) -> str:
    """Read a local image file and return a base64 data URI."""
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    ext = path.split(".")[-1].lower()
    return f"data:image/{ext};base64,{encoded}"

def analyze_image(image_path: str) -> ProductEntry:
    """Send one image to Gemini and get back a structured ProductEntry."""
    message = HumanMessage(content=[
        {"type": "text", "text": "Analyze this product photo and extract catalog details."},
        {"type": "image_url", "image_url": encode_image(image_path)},
    ])
    return structured_llm.invoke([message])

def build_catalog(image_folder: str, output_csv: str = "catalog.csv"):
    """Process every image in a folder and write results to a CSV file."""
    image_files = [
        f for f in os.listdir(image_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    entries = []
    for filename in image_files:
        full_path = os.path.join(image_folder, filename)
        print(f"Analyzing {filename}...")
        try:
            entry = analyze_image(full_path)
            entries.append(entry)
        except Exception as e:
            print(f"  Failed on {filename}: {e}")

    # Write all extracted rows to CSV
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(ProductEntry.model_fields.keys()))
        writer.writeheader()
        for entry in entries:
            writer.writerow(entry.model_dump())

    print(f"\nWrote {len(entries)} catalog entries to {output_csv}")

if __name__ == "__main__":
    build_catalog(image_folder="product_photos")  # put .jpg/.png files in this folder