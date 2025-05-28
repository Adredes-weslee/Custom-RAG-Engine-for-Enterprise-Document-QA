import sys
from pathlib import Path

import streamlit as st
import torch

# Add project root to path for utils import
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from langchain.vectorstores import FAISS
from sentence_transformers import SentenceTransformer

from config.config import *
from rag_engine.embeddings.faiss_index import load_faiss_index
from rag_engine.models.ollama_model import ollama_llm
from rag_engine.retrieval.document_store import create_docstore, load_documents
from rag_engine.retrieval.rag_chain import setup_rag_chain
from rag_engine.ui.streamlit_ui import setup_streamlit_ui
from utils.logging_setup import setup_logging

# Set up logging
logger = setup_logging()


def get_optimal_device():
    """Detect and use best available device for models"""
    if torch.cuda.is_available():
        logger.info("✅ GPU available - using CUDA acceleration")
        return "cuda"
    else:
        logger.info("⚠️ CPU-only mode - using CPU processing")
        return "cpu"


# Load FAISS indices
code_faiss_index_path = "./faiss_code_index.bin"
non_code_faiss_index_path = "./faiss_non_code_index.bin"
code_faiss_index = load_faiss_index(code_faiss_index_path)
non_code_faiss_index = load_faiss_index(non_code_faiss_index_path)

if code_faiss_index is None or non_code_faiss_index is None:
    st.error(
        "FAISS indices not found. Please ensure the FAISS index files exist at the specified paths."
    )
    st.stop()

# Load documents
code_docstore_path = "./code_docstore.json"
non_code_docstore_path = "./non_code_docstore.json"
code_documents = load_documents(code_docstore_path)
non_code_documents = load_documents(non_code_docstore_path)
code_docstore = create_docstore(code_documents)
non_code_docstore = create_docstore(non_code_documents)

# Ensure the index_to_docstore_id mapping covers all indices in the FAISS index
num_code_vectors = code_faiss_index.ntotal
num_non_code_vectors = non_code_faiss_index.ntotal
code_index_to_docstore_id = {i: i for i in range(len(code_documents))}
non_code_index_to_docstore_id = {i: i for i in range(len(non_code_documents))}
if num_code_vectors != len(code_index_to_docstore_id):
    st.warning(
        f"Mismatch between number of vectors in FAISS code index ({num_code_vectors}) and document store IDs ({len(code_index_to_docstore_id)}). Proceeding with available data."
    )
    code_index_to_docstore_id = {
        i: i % len(code_documents) for i in range(num_code_vectors)
    }
if num_non_code_vectors != len(non_code_index_to_docstore_id):
    st.warning(
        f"Mismatch between number of vectors in FAISS non-code index ({num_non_code_vectors}) and document store IDs ({len(non_code_index_to_docstore_id)}). Proceeding with available data."
    )
    non_code_index_to_docstore_id = {
        i: i % len(non_code_documents) for i in range(num_non_code_vectors)
    }

# Wrap the FAISS indices in LangChain-compatible vector stores
device = get_optimal_device()
model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
code_vector_store = FAISS(
    embedding_function=model.encode,
    index=code_faiss_index,
    docstore=code_docstore,
    index_to_docstore_id=code_index_to_docstore_id,
)
non_code_vector_store = FAISS(
    embedding_function=model.encode,
    index=non_code_faiss_index,
    docstore=non_code_docstore,
    index_to_docstore_id=non_code_index_to_docstore_id,
)

# Print final dimensionality of the query embeddings
query_embedding_dim = model.get_sentence_embedding_dimension()
logger.info(f"Final dimensionality of query embeddings: {query_embedding_dim}")

# Initialize Ollama LLM
llm = ollama_llm()

# Setup RAG chains
top_k = 15
code_rag_chain = setup_rag_chain(llm, code_vector_store, top_k)
non_code_rag_chain = setup_rag_chain(llm, non_code_vector_store, top_k)

# Setup Streamlit UI
setup_streamlit_ui(llm, code_rag_chain, non_code_rag_chain)
