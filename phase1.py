"""PDF Question Answering Pipeline using LangChain, FAISS, and HuggingFace APIs.

Steps:
1. Load PDF files.
2. Split documents into text chunks.
3. Embed chunks into dense vectors.
4. Store embeddings in FAISS vector database.
5. Setup a custom LLM QA Chain with retrieval.
"""

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA
from dotenv import load_dotenv, find_dotenv
import os

DATA_PATH = 'dataset'

def load_files(data_path: str):
    """
    Load PDF documents from a given directory.

    Args:
        data_path (str): Path to the folder containing PDF files.

    Returns:
        list: List of Document objects loaded from PDFs.
    """
    loader = DirectoryLoader(data_path, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents



documents = load_files(data_path=DATA_PATH)
print(f"Number of pages loaded: {len(documents)}")


def create_chunks(extracted_data):
    """
    Split documents into smaller chunks for better processing.

    Args:
        extracted_data (list): List of Document objects.

    Returns:
        list: List of text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=60)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks = create_chunks(extracted_data=documents)
print(f"Length of Text Chunks: {len(text_chunks)}")


def get_embedding_model():
    """
    Initialize the HuggingFace embedding model.

    Returns:
        HuggingFaceEmbeddings: Embedding model instance.
    """
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

embedding_model = get_embedding_model()
# print(embedding_model)

VECTOR_DATABASE_PATH = 'vectorstore/database_fs'


db = FAISS.from_documents(text_chunks, embedding_model)
db.save_local(VECTOR_DATABASE_PATH)



load_dotenv(find_dotenv())
HF_TOKEN = os.environ.get("HF_TOKEN")


HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"




def load_llm(huggingface_repo_id: str):
    """
    Load LLM from Hugging Face Inference Endpoint.

    Args:
        huggingface_repo_id (str): Model repository ID from HuggingFace.

    Returns:
        HuggingFaceEndpoint: LLM model instance.
    """
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        task="text-generation",  # <--- THIS IS MANDATORY NOW
        temperature=0.6,
        max_new_tokens= 512, 
    )
    return llm


CUSTOM_PROMPT_TEMPLATE = """
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

def set_custom_prompt(custom_prompt_template: str):
    """
    Create a custom PromptTemplate for QA.

    Args:
        custom_prompt_template (str): Prompt template string.

    Returns:
        PromptTemplate: Prompt template object.
    """
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

db = FAISS.load_local(VECTOR_DATABASE_PATH, embedding_model, allow_dangerous_deserialization=True)

qa_chain =  RetrievalQA.from_chain_type(
    llm = load_llm(HUGGINGFACE_REPO_ID),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)



user_query = input("Ask your question: ")
response = qa_chain.invoke({'query': user_query})

print(response["result"])
print("\n::::> SOURCE DOCUMENTS")
for doc in response["source_documents"]:
    print(doc.metadata.get("source", "Unknown source"))
