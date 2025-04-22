from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from typing import Tuple

def load_models() -> Tuple[SentenceTransformer, AutoTokenizer, AutoModel]:
    """
    Load pre-trained models for sentence and code embeddings.

    Returns:
        Tuple[SentenceTransformer, AutoTokenizer, AutoModel]: Loaded models.
    """
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    code_tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")
    code_model = AutoModel.from_pretrained("microsoft/graphcodebert-base")
    return sentence_model, code_tokenizer, code_model