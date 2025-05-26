from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from typing import Tuple, Optional
from utils.logging_setup import setup_logging

# Set up logging
logger = setup_logging()

def load_models(use_safetensors: bool = True) -> Tuple[SentenceTransformer, AutoTokenizer, AutoModel]:
    """
    Load pre-trained models for sentence and code embeddings.
    
    Args:
        use_safetensors (bool): If True, prefer models that support safetensors format

    Returns:
        Tuple[SentenceTransformer, AutoTokenizer, AutoModel]: Loaded models.
    """
    try:
        if use_safetensors:
            # Use models that are known to work well with safetensors
            sentence_model = SentenceTransformer('all-MiniLM-L6-v2', trust_remote_code=False)
            # Use a lighter code model that works better with safetensors
            code_tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base", trust_remote_code=False)
            code_model = AutoModel.from_pretrained("microsoft/codebert-base", trust_remote_code=False)
        else:
            # Original models
            sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            code_tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")
            code_model = AutoModel.from_pretrained("microsoft/graphcodebert-base")
            
        logger.info(f"Successfully loaded models (safetensors mode: {use_safetensors})")
        return sentence_model, code_tokenizer, code_model
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        if use_safetensors:
            logger.info("Falling back to original models...")
            return load_models(use_safetensors=False)
        raise