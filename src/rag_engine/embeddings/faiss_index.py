import faiss
import os
import sys
from pathlib import Path
from typing import Optional
import numpy as np

# Add project root to path for utils import
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.logging_setup import setup_logging


# Set up logging
logger = setup_logging()

def create_faiss_index(embeddings: np.ndarray, target_dim: int) -> faiss.IndexFlatL2:
    """
    Create a FAISS index for the given embeddings.

    Args:
        embeddings (np.ndarray): Array of embeddings.
        target_dim (int): Target dimensionality.

    Returns:
        faiss.IndexFlatL2: FAISS index.
    """
    index = faiss.IndexFlatL2(target_dim)
    index.add(embeddings)
    return index

def save_faiss_index(index: faiss.IndexFlatL2, file_path: str) -> None:
    """
    Save the FAISS index to disk.

    Args:
        index (faiss.IndexFlatL2): FAISS index.
        file_path (str): Path to save the index.
    """
    faiss.write_index(index, file_path)
    logger.info(f"FAISS index saved to disk as '{file_path}'.")

def load_faiss_index(file_path: str) -> Optional[faiss.IndexFlatL2]:
    """
    Load the FAISS index from disk.

    Args:
        file_path (str): Path to the FAISS index file.

    Returns:
        Optional[faiss.IndexFlatL2]: Loaded FAISS index or None if the file does not exist.
    """
    if os.path.exists(file_path):
        index = faiss.read_index(file_path)
        logger.info("FAISS index loaded from disk.")
        return index
    else:
        logger.error(f"FAISS index file {file_path} does not exist.")
        return None