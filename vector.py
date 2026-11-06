from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from langchain_community.document_loaders import PyPDFLoader
from huggingface_hub import login
import os
import sys
import traceback
import chromadb

from utils import log_err, get_file_name

login(os.getenv("HF_TOKEN"))

def collection_exists(pdf_path: str) -> bool:
    client = chromadb.PersistentClient("./chroma_db")
    collection_name = get_file_name(pdf_path)
    
    for coll in client.list_collections():
        if coll.name == collection_name:
            return True
    return False

def create_vector_store(pdf_path: str) -> bool:
    """Creates the Vector DB for the attached pdf

    Args:
        pdf_path (str): pdf path

    Returns:
        bool: acknowledgement
    """
    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 200)

        chunks = text_splitter.split_documents(documents)
        embeddings = HuggingFaceEmbeddings(model_name = "BAAI/bge-small-en-v1.5")

        Chroma.from_documents(
            documents = chunks,
            collection_name = get_file_name(pdf_path),
            embedding = embeddings,
            persist_directory = "./chroma_db"
        )

        return True
    except Exception as err:
        log_err(err)
        traceback.print_exc()
        return False

def load_vector_store(pdf_path: str) -> Chroma:
    """Loads vector database

    Args:
        pdf_path (str): pdf path

    Returns:
        Chroma: Chroma object
    """
    try:
        embeddings = HuggingFaceEmbeddings(model_name = "BAAI/bge-small-en-v1.5")
        vector_store = Chroma(
            collection_name = get_file_name(pdf_path),
            persist_directory = "./chroma_db",
            embedding_function = embeddings
        )

        return vector_store
    except Exception as err:
        log_err(err)

def perform_similarity_search(pdf_path: str, prompt: str, k:  int = 3) -> list[str]:
    try:
        vector_store = load_vector_store(pdf_path)
        results = vector_store.similarity_search(query = prompt, k = 3)

        context = []
        for cont in results:
            context.append(cont.page_content)

        return context
    except Exception as err:
        traceback.print_exc()
    