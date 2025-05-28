import sys
from pathlib import Path
from typing import Tuple

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

# Add project root to path for utils import
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.logging_setup import setup_logging

# Set up logging
logger = setup_logging()


def get_optimal_device():
    """Detect and use best available device"""
    if torch.cuda.is_available():
        device = "cuda"
        logger.info("✅ GPU available - using CUDA acceleration for models")
        return device
    else:
        device = "cpu"
        logger.info("⚠️ CPU-only mode - using CPU for model inference")
        return device


def load_models(
    use_safetensors: bool = True,
) -> Tuple[SentenceTransformer, AutoTokenizer, AutoModel]:
    """
    Load pre-trained models for sentence and code embeddings with GPU acceleration.

    Args:
        use_safetensors (bool): If True, prefer models that support safetensors format

    Returns:
        Tuple[SentenceTransformer, AutoTokenizer, AutoModel]: Loaded models optimized for available hardware.
    """
    device = get_optimal_device()

    try:
        if use_safetensors:
            # Use models that are known to work well with safetensors
            sentence_model = SentenceTransformer(
                "all-MiniLM-L6-v2", trust_remote_code=False, device=device
            )
            # Use a lighter code model that works better with safetensors
            code_tokenizer = AutoTokenizer.from_pretrained(
                "microsoft/codebert-base", trust_remote_code=False
            )
            code_model = AutoModel.from_pretrained(
                "microsoft/codebert-base", trust_remote_code=False
            ).to(device)
        else:
            # Original models
            sentence_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
            code_tokenizer = AutoTokenizer.from_pretrained(
                "microsoft/graphcodebert-base"
            )
            code_model = AutoModel.from_pretrained("microsoft/graphcodebert-base").to(
                device
            )

        logger.info(
            f"🚀 Successfully loaded models on {device.upper()} (safetensors mode: {use_safetensors})"
        )
        return sentence_model, code_tokenizer, code_model

    except Exception as e:
        logger.error(f"Error loading models: {e}")
        if use_safetensors:
            logger.info("Falling back to original models...")
            return load_models(use_safetensors=False)
        raise
