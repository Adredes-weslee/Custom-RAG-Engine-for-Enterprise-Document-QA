import sys
from pathlib import Path

# Add project root to path for utils import
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from langchain_ollama.llms import OllamaLLM

from utils.logging_setup import setup_logging
from utils.model_config import (
    get_fallback_model,
    get_judge_model,
    get_primary_model,
)

logger = setup_logging()


def load_ollama_model(model_type: str = "primary") -> OllamaLLM:
    """
    Load and initialize the Ollama model with environment-aware model selection.

    Args:
        model_type (str): Type of model to load ('primary', 'judge', or 'fallback')

    Returns:
        OllamaLLM: Initialized language model instance.
    """
    if model_type == "primary":
        model_name = get_primary_model()
    elif model_type == "judge":
        model_name = get_judge_model()
    elif model_type == "fallback":
        model_name = get_fallback_model()
    else:
        logger.warning(f"Unknown model type '{model_type}', using primary model")
        model_name = get_primary_model()

    logger.info(f"🤖 Loading Ollama {model_type} model: {model_name}")

    try:
        return OllamaLLM(model=model_name)
    except Exception as e:
        logger.error(f"❌ Failed to load {model_type} model '{model_name}': {e}")
        if model_type != "fallback":
            logger.info("🔄 Attempting to load fallback model...")
            return OllamaLLM(model=get_fallback_model())
        else:
            raise


def ollama_llm(model_type: str = "primary") -> OllamaLLM:
    """
    Return the Ollama LLM instance with environment-aware model selection.

    Args:
        model_type (str): Type of model to load ('primary', 'judge', or 'fallback')

    Returns:
        OllamaLLM: Language model instance.
    """
    return load_ollama_model(model_type)


def get_primary_llm() -> OllamaLLM:
    """Get the primary RAG model instance."""
    return ollama_llm("primary")


def get_judge_llm() -> OllamaLLM:
    """Get the evaluation/judge model instance."""
    return ollama_llm("judge")
