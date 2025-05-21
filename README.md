# RAG-Project-PDF-Chat
This is a Rag based project where we can chat with PDF by making a knowledge base using vector database and LLM for accurate results.

# PDF Question Answering Pipeline 🧠📄

This project builds a **Question Answering (QA)** system that can **ingest PDF documents, process them into searchable vector chunks**, and respond to natural language queries using **retrieval-augmented generation (RAG)**. It uses:

- 🦜 LangChain for pipeline orchestration
- 🤗 HuggingFace for embeddings & LLMs
- 🔍 FAISS for vector search
- 📄 PyPDFLoader for document loading

---

## 🔧 Features

- Load multiple PDFs from a directory
- Split documents into overlapping text chunks
- Convert text chunks into dense vector embeddings
- Store vectors in a FAISS vector database
- Use a custom prompt template for human-like responses
- Retrieve top-k relevant chunks and answer user questions using a HuggingFace-hosted LLM

---

## 🗂 Directory Structure

project/
│
├── dataset/ # Folder containing PDF files ## you can add more pdf files here 
├── vectorstore/ # Saved FAISS vector DB
├── Phase1.py # This code you can run in terminal without the streamlit app
├── Phase2.py # This code you can run as app in localhost with interface
├── .env # Your HuggingFace API token
└── README.md # This file


---

## 🚀 Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/your-repo/pdf-qa-pipeline.git
cd pdf-qa-pipeline
pip install -r requirements.txt

### make sure you change your Hugging face token here for connecting with LLM

##To run with streamit 
'''pip install streamlit'''
streamlit run phase2.py


