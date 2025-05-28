import os
import sys
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import torch

# Add project root to path for utils import
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.logging_setup import setup_logging

# Set up logging
logger = setup_logging()


def get_optimal_device():
    """Detect and use best available device"""
    if torch.cuda.is_available():
        logger.info("✅ GPU available - using CUDA acceleration")
        return "cuda"
    else:
        logger.info("⚠️ CPU-only mode - using CPU processing")
        return "cpu"


def create_faiss_index(embeddings: np.ndarray, target_dim: int) -> faiss.IndexFlatL2:
    """
    Create a FAISS index for the given embeddings with GPU acceleration if available.

    Args:
        embeddings (np.ndarray): Array of embeddings.
        target_dim (int): Target dimensionality.

    Returns:
        faiss.IndexFlatL2: FAISS index.
    """
    device = get_optimal_device()

    # Create the base index
    index = faiss.IndexFlatL2(target_dim)

    # Move to GPU if available
    if device == "cuda" and faiss.get_num_gpus() > 0:
        try:
            # Create GPU resource
            gpu_resource = faiss.StandardGpuResources()
            # Move index to GPU
            gpu_index = faiss.index_cpu_to_gpu(gpu_resource, 0, index)
            gpu_index.add(embeddings)
            # Move back to CPU for storage/return
            index = faiss.index_gpu_to_cpu(gpu_index)
            logger.info(
                f"✅ FAISS index created with GPU acceleration ({embeddings.shape[0]} vectors)"
            )
        except Exception as e:
            logger.warning(f"⚠️ GPU acceleration failed, falling back to CPU: {e}")
            index.add(embeddings)
    else:
        index.add(embeddings)
        logger.info(f"📊 FAISS index created with CPU ({embeddings.shape[0]} vectors)")

    return index


def save_faiss_index(index: faiss.IndexFlatL2, file_path: str) -> None:
    """
    Save a FAISS index to disk with GPU-aware handling.

    Args:
        index (faiss.IndexFlatL2): FAISS index to save.
        file_path (str): Path to save the index.
    """
    try:
        # Ensure index is on CPU before saving
        if hasattr(index, "device") and index.device != -1:
            # If it's a GPU index, move to CPU first
            cpu_index = faiss.index_gpu_to_cpu(index)
            faiss.write_index(cpu_index, file_path)
        else:
            faiss.write_index(index, file_path)

        logger.info(f"💾 FAISS index saved to: {file_path}")
    except Exception as e:
        logger.error(f"❌ Failed to save FAISS index: {e}")
        raise


def load_faiss_index(file_path: str) -> Optional[faiss.IndexFlatL2]:
    """
    Load the FAISS index from disk with optional GPU acceleration.

    Args:
        file_path (str): Path to the FAISS index file.

    Returns:
        Optional[faiss.IndexFlatL2]: Loaded FAISS index or None if the file does not exist.
    """
    if os.path.exists(file_path):
        try:
            # Load index from disk (always CPU initially)
            index = faiss.read_index(file_path)

            # Optionally move to GPU for faster operations
            device = get_optimal_device()
            if device == "cuda" and faiss.get_num_gpus() > 0:
                try:
                    gpu_resource = faiss.StandardGpuResources()
                    gpu_index = faiss.index_cpu_to_gpu(gpu_resource, 0, index)
                    logger.info(f"🚀 FAISS index loaded and moved to GPU: {file_path}")
                    # Return GPU index wrapped for compatibility
                    return gpu_index
                except Exception as e:
                    logger.warning(
                        f"⚠️ GPU acceleration failed during load, using CPU: {e}"
                    )

            logger.info(f"📂 FAISS index loaded from disk (CPU): {file_path}")
            return index

        except Exception as e:
            logger.error(f"❌ Failed to load FAISS index from {file_path}: {e}")
            return None
    else:
        logger.error(f"FAISS index file {file_path} does not exist.")
        return None
