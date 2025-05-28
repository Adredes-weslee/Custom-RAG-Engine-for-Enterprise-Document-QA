from typing import List

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def generate_code_embeddings(
    texts: List[str], code_tokenizer: AutoTokenizer, code_model: AutoModel
) -> List[np.ndarray]:
    """
    Generate embeddings for Python files using GraphCodeBERT with GPU acceleration.

    Args:
        texts (List[str]): List of texts to generate embeddings for.
        code_tokenizer (AutoTokenizer): Tokenizer for code model.
        code_model (AutoModel): Pre-trained code model.

    Returns:
        List[np.ndarray]: List of embeddings.
    """
    embeddings = []
    device = next(code_model.parameters()).device  # Get the device of the model

    for text in tqdm(texts, desc="Generating code embeddings"):
        inputs = code_tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        # Move inputs to the same device as the model
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = code_model(**inputs)
        embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy())
    return embeddings


def generate_sentence_embeddings(
    texts: List[str], sentence_model: SentenceTransformer
) -> List[np.ndarray]:
    """
    Generate embeddings for Jupyter notebooks using SentenceTransformer.

    Args:
        texts (List[str]): List of texts to generate embeddings for.
        sentence_model (SentenceTransformer): Pre-trained sentence model.

    Returns:
        List[np.ndarray]: List of embeddings.
    """
    return sentence_model.encode(texts, show_progress_bar=True)


def project_embeddings(embeddings: np.ndarray, target_dim: int) -> np.ndarray:
    """
    Project embeddings to a common dimensionality.

    Args:
        embeddings (np.ndarray): Array of embeddings.
        target_dim (int): Target dimensionality.

    Returns:
        np.ndarray: Projected embeddings.
    """
    projection_matrix = np.random.randn(embeddings.shape[1], target_dim)
    projected_embeddings = np.dot(embeddings, projection_matrix)
    return projected_embeddings
