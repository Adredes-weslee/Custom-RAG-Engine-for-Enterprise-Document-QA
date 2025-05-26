import os
import json
import sys
from pathlib import Path
from utils.logging_setup import setup_logging
from typing import List, Dict
from langchain.docstore import InMemoryDocstore
from langchain.docstore.document import Document

# Set up logging
logger = setup_logging()

def load_documents(docstore_path: str) -> List[Dict]:
    """
    Load documents from a JSON file.

    Args:
        docstore_path (str): Path to the document store JSON file.

    Returns:
        List[Dict]: List of documents loaded from the file.
    """
    if os.path.exists(docstore_path):
        with open(docstore_path, 'r') as f:
            documents = json.load(f)
        logger.info("Documents loaded from 'docstore.json'.")
        return documents
    else:
        raise FileNotFoundError("Document store file not found. Please ensure 'docstore.json' exists.")

def create_docstore(documents: List[Dict]) -> InMemoryDocstore:
    """
    Create an in-memory document store from a list of documents.

    Args:
        documents (List[Dict]): List of documents.

    Returns:
        InMemoryDocstore: In-memory document store.
    """
    return InMemoryDocstore({i: Document(page_content=doc["content"], metadata={"source": doc["source"]}) for i, doc in enumerate(documents)})