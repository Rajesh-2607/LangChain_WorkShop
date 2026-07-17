# 🦜🔗 LangChain Workshop: From Basics to Capstone Projects

![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)
![LangChain](https://img.shields.io/badge/LangChain-Integration-green)
![Gemini](https://img.shields.io/badge/Google%20GenAI-Gemini%20Pro-orange)

Welcome to the **LangChain Workshop** repository! This comprehensive guide and codebase are designed to take you from the fundamental concepts of Large Language Models (LLMs) to building advanced, production-ready AI agents and Retrieval-Augmented Generation (RAG) pipelines using LangChain and Google's Gemini models.

## 📖 Table of Contents
1. [Project Overview](#project-overview)
2. [Prerequisites](#prerequisites)
3. [Repository Structure & Curriculum](#repository-structure--curriculum)
4. [Deep Dive: Embeddings & Vector Stores](#deep-dive-embeddings--vector-stores)
5. [Installation and Setup](#installation-and-setup)
6. [Running the Code](#running-the-code)
7. [Capstone Projects](#capstone-projects)
8. [Author](#author)

---

## 🚀 Project Overview

This repository acts as a step-by-step interactive workshop. Rather than just reading documentation, you will execute Python scripts that demonstrate key LangChain capabilities in isolation before combining them into complex applications. 

The primary LLM provider used throughout this workshop is **Google Generative AI**, specifically leveraging the `gemini-pro` model for text generation and `gemini-embedding-001` for vector representations.

---

## ⚠️ Prerequisites

Before cloning and running this repository, ensure you have the following:
* **Python 3.9 or higher** installed on your system.
* A basic understanding of Python programming and object-oriented concepts.
* A **Google Gemini API Key**. You can obtain this from Google AI Studio.
* Familiarity with virtual environments (recommended).

---

## 📂 Repository Structure & Curriculum

The files are sequentially numbered to provide a logical learning path. 

### Part 1: Core Fundamentals
* `01_first_call.py`: Initializing the LLM and making a basic inference request.
* `02_messages.py`: Understanding different message types (System, Human, AI) in chat models.
* `03_streaming.py`: Handling streaming responses for lower perceived latency.
* `04_batch.py`: Executing batch prompts for efficient processing.
* `05_async.py`: Utilizing asynchronous calls for non-blocking execution.

### Part 2: Prompt Engineering & Chains
* `06_prompt_templates.py`: Creating dynamic prompts with input variables.
* `07_few_shot.py`: Providing examples to the LLM to guide its output format and tone.
* `08_lcel_chain.py`: Introduction to LangChain Expression Language (LCEL) to pipe components together (`Prompt | LLM | Parser`).
* `09_output_parsers.py`: Forcing the LLM to return data in specific formats.

### Part 3: Advanced LangChain Features
* `10_token_counting.py`: Estimating costs and managing context windows.
* `11_structured_output.py`: Extracting precise JSON or Pydantic models from text.
* `12_streaming_structured_output.py`: Advanced streaming of structured data.
* `13_tool_calling.py`: Giving the LLM the ability to execute external functions.
* `14_google_search_grounding.py`: Using Google Search as a tool for real-time data retrieval.
* `15_code_execution.py`: Allowing the agent to write and execute code.

### Part 4: Multimodality & Context
* `16_image_input.py`: Passing images to Gemini Vision models.
* `17_video_input.py`: Video processing capabilities.
* `18_memory.py`: Implementing conversational memory so the LLM remembers past interactions.

---

## 🔍 Deep Dive: Embeddings & Vector Stores

Scripts `19` and `20` form the backbone of our Retrieval-Augmented Generation (RAG) architecture.

### `19_embeddings.py`
This script demonstrates how to convert text into numerical vectors using `GoogleGenerativeAIEmbeddings` (specifically the `gemini-embedding-001` model). 
* **Batch Embedding:** Shows how to process up to 100 strings simultaneously to save API calls.
* **Task Types:** Explains how to tune embeddings for specific use cases by passing `task_type="RETRIEVAL_QUERY"` or `task_type="RETRIEVAL_DOCUMENT"`.
* **Similarity Search:** Uses `sklearn.metrics.pairwise.cosine_similarity` to mathematically calculate the distance between a query vector ("What is the capital of France?") and document vectors.

### `20_vector_store.py`
Builds upon the embeddings by introducing a Vector Database to persist and efficiently query those mathematical representations, setting the stage for the capstone projects.

---

## 🛠️ Installation and Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Rajesh-2607/LangChain_WorkShop.git](https://github.com/Rajesh-2607/LangChain_WorkShop.git)
   cd LangChain_WorkShop