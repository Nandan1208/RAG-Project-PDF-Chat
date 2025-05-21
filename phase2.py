# app.py

import os
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA

# ------------- Load environment variables ---------------
load_dotenv(find_dotenv())
HF_TOKEN = os.getenv("HF_TOKEN")


# ------------- Constants -----------------
DATA_PATH = 'dataset'
VECTOR_DATABASE_PATH = 'vectorstore/database_fs'
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

# ------------- Functions ------------------

def load_files(data_path: str):
    loader = DirectoryLoader(data_path, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=60)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

def load_llm(huggingface_repo_id: str):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        task="text-generation",  # Mandatory
        temperature=0.6,
        max_new_tokens=512,
    )
    return llm

def set_custom_prompt(custom_prompt_template: str):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

# ------------- Streamlit App ------------------

st.set_page_config(page_title="PDF Q&A App", layout="wide")
st.title("📚 Ask Questions from your PDFs!")

# Sidebar
with st.sidebar:
    st.header("Setup")
    if st.button("Load and Process Documents"):
        with st.spinner("Loading documents..."):
            documents = load_files(DATA_PATH)
            st.session_state.documents = documents
            st.success(f"Loaded {len(documents)} pages.")

        with st.spinner("Splitting into chunks..."):
            text_chunks = create_chunks(documents)
            st.session_state.text_chunks = text_chunks
            st.success(f"Created {len(text_chunks)} chunks.")

        with st.spinner("Creating embeddings and FAISS database..."):
            embedding_model = get_embedding_model()
            db = FAISS.from_documents(text_chunks, embedding_model)
            db.save_local(VECTOR_DATABASE_PATH)
            st.session_state.db = db
            st.success("Database created and saved!")

# Load Vector Database
if "db" not in st.session_state:
    if os.path.exists(VECTOR_DATABASE_PATH):
        with st.spinner("Loading vector database..."):
            embedding_model = get_embedding_model()
            db = FAISS.load_local(VECTOR_DATABASE_PATH, embedding_model, allow_dangerous_deserialization=True)
            st.session_state.db = db
    else:
        st.warning("Please load and process documents first from the sidebar.")

# Set up QA Chain
if "db" in st.session_state:
    qa_chain = RetrievalQA.from_chain_type(
        llm = load_llm(HUGGINGFACE_REPO_ID),
        chain_type="stuff",
        retriever=st.session_state.db.as_retriever(search_kwargs={'k': 3}),
        return_source_documents=True,
        chain_type_kwargs={
            'prompt': set_custom_prompt("""
    You are a helpful and friendly assistant that answers questions strictly based on the provided context.

    Instructions:
    - Start the response with a short, friendly greeting like "Hello!", "Hi there!", or "Hey!".
    - Vary the greeting naturally to make responses feel less robotic.
    - After the greeting, immediately answer the user's question based only on the given context.
    - If the answer cannot be found in the context, reply: "I don't know based on the provided information."
    - Avoid adding any external knowledge, personal opinions, or fabricated information.
    - Keep the answer clear, factual, and concise.

    Here is the context:
    {context}

    Here is the user's question:
    {question}

    Respond naturally:
    """
)
        }
    )

    # Main input area
    user_query = st.text_input("Ask your question:")

    if user_query:
        with st.spinner("Thinking..."):
            response = qa_chain.invoke({'query': user_query})
        
        st.subheader("📝 Answer")
        st.write(response["result"])

        st.subheader("📄 Source Documents")
        for i, doc in enumerate(response["source_documents"], 1):
            st.markdown(f"**Document {i}:** {doc.metadata.get('source', 'Unknown source')}")
else:
    st.info("Load or create the database first to start asking questions.")

