from utils.logging_setup import setup_logging
from langchain.chat_models import AzureChatOpenAI
from langchain.schema import HumanMessage
from langchain_ollama.llms import OllamaLLM

# Set up logging
logger = setup_logging()

def evaluate_answer_with_ollama(answer: str, question: str, ollama_llm: OllamaLLM) -> str:
    """
    Use Ollama Gemma2 to evaluate the quality of the answer based on accuracy, relevance, and completeness.
    Provide feedback on how well the answer addresses the user's question.

    Args:
        answer (str): The answer produced by the RAG system.
        question (str): The original user question.
        ollama_llm (OllamaLLM): The Ollama language model instance.

    Returns:
        str: Evaluation feedback including scores for relevance and correctness.
    """
    evaluation_prompt = f"""
    You are an AI assistant. The following answer was produced for a user's question. Please evaluate its quality, considering:

    1. **Relevance**: Does the answer directly address the user's question?
    2. **Correctness**: Is the information factually accurate and does it provide a solution?
    3. **Completeness**: Is the answer thorough enough to resolve the query, or does it need more detail?

    Rate the answer on a scale of 1 to 5 for each category, and explain your ratings.

    User's Question: {question}
    System's Answer: {answer}

    Please provide your evaluation:
    - Rating for relevance (1-5):
    - Explanation for relevance rating:
    - Rating for correctness (1-5):
    - Explanation for correctness rating:
    - Rating for completeness (1-5):
    - Explanation for completeness rating:
    """
    
    response = ollama_llm.invoke([HumanMessage(content=evaluation_prompt)])
    evaluation_feedback = response.strip()
    
    # Log the LLM output to the terminal for debugging
    logger.info(f"Evaluation Prompt: {evaluation_prompt}")
    logger.info(f"LLM Response: {evaluation_feedback}")
    
    return evaluation_feedback

##### Uncomment if you wish to use the AzureChatOpenAI model instead of the OllamaLLM model. #####

# def evaluate_answer_with_azure(answer: str, question: str, azure_llm: AzureChatOpenAI) -> str:
#     """
#     Use Azure OpenAI to evaluate the quality of the answer based on accuracy, relevance, and completeness.
#     Provide feedback on how well the answer addresses the user's question.

#     Args:
#         answer (str): The answer produced by the RAG system.
#         question (str): The original user question.
#         azure_llm (AzureChatOpenAI): The Azure OpenAI language model instance.

#     Returns:
#         str: Evaluation feedback including scores for relevance and correctness.
#     """
#     evaluation_prompt = f"""
#     You are an AI assistant. The following answer was produced for a user's question. Please evaluate its quality, considering:

#     1. **Relevance**: Does the answer directly address the user's question?
#     2. **Correctness**: Is the information factually accurate and does it provide a solution?
#     3. **Completeness**: Is the answer thorough enough to resolve the query, or does it need more detail?

#     Rate the answer on a scale of 1 to 5 for each category, and explain your ratings.

#     User's Question: {question}
#     System's Answer: {answer}

#     Please provide your evaluation:
#     - Rating for relevance (1-5):
#     - Explanation for relevance rating:
#     - Rating for correctness (1-5):
#     - Explanation for correctness rating:
#     - Rating for completeness (1-5):
#     - Explanation for completeness rating:
#     """
    
#     response = azure_llm.invoke([HumanMessage(content=evaluation_prompt)])
#     evaluation_feedback = response.content.strip()
    
#     # Log the LLM output to the terminal for debugging
#     logger.info(f"Evaluation Prompt: {evaluation_prompt}")
#     logger.info(f"LLM Response: {evaluation_feedback}")
    
#     return evaluation_feedback