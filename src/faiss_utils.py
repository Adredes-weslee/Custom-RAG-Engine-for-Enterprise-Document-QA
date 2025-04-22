import os
import faiss
from logging_setup import logger

def load_faiss_index(file_path: str) -> faiss.Index:
    """
    Load FAISS index from disk.

    Args:
        file_path (str): Path to the FAISS index file.

    Returns:
        faiss.Index: Loaded FAISS index, or None if the file does not exist.
    """
    if os.path.exists(file_path):
        logger.info(f"FAISS index file found at {file_path}.")
        index = faiss.read_index(file_path)
        logger.info("FAISS index loaded from disk.")
        return index
    else:
        logger.error(f"FAISS index file {file_path} does not exist.")
        return None