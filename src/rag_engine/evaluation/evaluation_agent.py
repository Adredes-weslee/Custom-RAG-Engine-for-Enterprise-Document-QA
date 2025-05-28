import sys
from pathlib import Path

# Add project root to path for utils import
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from langchain_ollama.llms import OllamaLLM

from utils.logging_setup import setup_logging
from utils.model_config import get_judge_model

# Set up logging
logger = setup_logging()


def evaluate_answer_with_ollama(
    answer: str, question: str, ollama_llm: OllamaLLM = None
) -> str:
    """
    Use Ollama judge model to evaluate the quality of the answer based on accuracy, relevance, and completeness.
    Provide feedback on how well the answer addresses the user's question.

    Args:
        answer (str): The answer produced by the RAG system.
        question (str): The original user question.
        ollama_llm (OllamaLLM, optional): The Ollama language model instance. If None, uses judge model.

    Returns:
        str: Evaluation feedback including scores for relevance and correctness.
    """
    # Use provided LLM or create judge model instance
    if ollama_llm is None:
        judge_model = get_judge_model()
        ollama_llm = OllamaLLM(model=judge_model)
        logger.info(f"🔍 Using judge model for evaluation: {judge_model}")

    evaluation_prompt = f"""You are an AI assistant evaluating the quality of an answer. Please be concise and rate 1-5 for each category:

User's Question: {question}
System's Answer: {answer}

Evaluation:
1. Relevance (1-5): How well does the answer address the question?
2. Correctness (1-5): Is the information accurate?
3. Completeness (1-5): Is the answer thorough enough?

Format your response as:
Relevance: X/5 - brief explanation
Correctness: X/5 - brief explanation  
Completeness: X/5 - brief explanation"""

    try:
        # ✅ Fixed: Direct string input instead of HumanMessage
        response = ollama_llm.invoke(evaluation_prompt)
        evaluation_feedback = (
            response.strip() if hasattr(response, "strip") else str(response).strip()
        )

        # Log for debugging (shorter logs)
        logger.info(f"📊 Evaluation completed for question: {question[:50]}...")

        return evaluation_feedback

    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        return f"Evaluation failed: {str(e)}"
