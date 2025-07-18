import sys
from pathlib import Path

from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

# Add project root to path for utils import
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.logging_setup import setup_logging
from utils.model_config import get_primary_model

# Set up logging
logger = setup_logging()


def handle_code_query(
    user_question: str, code_rag_chain: ConversationalRetrievalChain, llm: OllamaLLM
) -> tuple:
    """
    Handle the user's code-related query by comparing it with the code in the RAG database.

    Args:
        user_question (str): The user's question with code.
        code_rag_chain (ConversationalRetrievalChain): RAG chain instance for code-related queries.
        llm (OllamaLLM): Language model instance for initial evaluation.

    Returns:
        tuple: Answer, retrieved documents, and reasoning for the suggestion.
    """
    # Directly process the question since it's code-related
    response = code_rag_chain({"question": user_question})
    answer = response["answer"]
    retrieved_docs = response["source_documents"]

    # Log the query and the LLM output to the terminal for debugging
    logger.info(f"Original Query: {user_question}")
    logger.info(f"Ollama LLM Response for Code Query: {answer}")

    return answer, retrieved_docs, "Code RAG chain used"


def handle_non_code_query(
    user_question: str, non_code_rag_chain: ConversationalRetrievalChain, llm: OllamaLLM
) -> tuple:
    """
    Handle non-code related queries, enhancing the query for better retrieval.

    Args:
        user_question (str): The user's theory or concept-related question.
        non_code_rag_chain (ConversationalRetrievalChain): RAG chain instance for non-code-related queries.
        llm (OllamaLLM): Language model instance.

    Returns:
        tuple: Answer, retrieved documents, and reasoning for the response.
    """
    try:
        # Step 1: Enhance the user question
        enhanced_question = enhance_query_with_llm(user_question, llm)

        # Log the enhanced query to the terminal for debugging
        logger.info(f"Enhanced Query: {enhanced_question}")

        # Step 2: Retrieve the answer from the non-code RAG chain
        response = non_code_rag_chain({"question": enhanced_question})
        answer = response["answer"]
        retrieved_docs = response["source_documents"]

        # Log the query and the LLM output to the terminal for debugging
        logger.info(f"Original Query: {user_question}")
        logger.info(f"Ollama LLM Response for Non-Code Query: {answer}")

        return answer, retrieved_docs, "Non-code RAG chain used"
    except Exception as e:
        logger.error(f"Error in handle_non_code_query: {e}")
        raise


def enhance_query_with_llm(query: str, llm: OllamaLLM) -> str:
    """
    Enhance the query using an LLM by providing additional context, improving clarity, and expanding with relevant terms.
    This prepares the query for accurate retrieval from the database, especially when determining if the user's code is good
    or if they need to improve.

    Args:
        query (str): The original user query.
        llm (OllamaLLM): The language model instance.

    Returns:
        str: The enhanced query.
    """
    try:
        logger.info(f"Original Query: {query}")

        template = """The following query needs enhancement to be suitable for retrieval. Please add context, improve clarity, 
        and expand it with relevant terms to help the system determine whether the code is of high quality compared to the repository.

        Original Query: {query}

        Enhanced Query:"""

        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | llm

        response = chain.invoke({"query": query})
        enhanced_query = response.strip()

        # Log the enhanced query to the terminal for debugging
        logger.info(f"Enhanced Query: {enhanced_query}")

        return enhanced_query
    except Exception as e:
        logger.error(f"Error in enhance_query_with_llm: {e}")
        raise


def determine_approach(user_question: str, llm: OllamaLLM) -> str:
    """
    Use the LLM to determine if the user's question should be directed to the code RAG chain (for code-related queries)
    or the non-code RAG chain (for theory or general questions about machine learning/AI).

    Args:
        user_question (str): The user's enhanced question.
        llm (OllamaLLM): Language model instance.

    Returns:
        str: Reasoning for whether the question should be routed to the code or non-code RAG chain.
    """
    reasoning_prompt = f"""
    You are an AI assistant tasked with determining whether the user's question should be answered using the code RAG chain 
    (for retrieving relevant code examples and assessing code quality) or the non-code RAG chain (for answering theoretical 
    machine learning/AI questions).

    If the question is about code quality, snippets, errors, or implementation details, use the code RAG chain.
    If the question is more theoretical or about general concepts, use the non-code RAG chain.

    User's Question: "{user_question}"

    Which approach should be used: code RAG chain or non-code RAG chain? Provide your reasoning.
    """

    template = """Question: {question}
    Answer: Let's think step by step."""

    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm

    response = chain.invoke({"question": reasoning_prompt})
    reasoning = response.strip()

    # Log the query and the LLM output to the terminal for debugging
    logger.info(f"Original Query: {user_question}")
    logger.info(f"Reasoning: {reasoning}")

    return reasoning


def handle_question(
    user_question: str,
    code_rag_chain: ConversationalRetrievalChain,
    non_code_rag_chain: ConversationalRetrievalChain,
    llm: OllamaLLM = None,
) -> tuple:
    """
    Process the user's question by enhancing it, determining the appropriate approach, and retrieving the answer
    from either the code or non-code RAG chain.

    Args:
        user_question (str): The user's question.
        code_rag_chain (ConversationalRetrievalChain): RAG chain instance for code-related queries.
        non_code_rag_chain (ConversationalRetrievalChain): RAG chain instance for non-code-related queries.
        llm (OllamaLLM, optional): Language model instance. If None, uses primary model.

    Returns:
        tuple: Answer, retrieved documents, and reasoning for the chosen approach.
    """
    # Use provided LLM or create primary model instance
    if llm is None:
        primary_model = get_primary_model()
        llm = OllamaLLM(model=primary_model)
        logger.info(f"🔧 Using primary model for question handling: {primary_model}")

    # Step 1: Determine which RAG chain to use
    reasoning = determine_approach(user_question, llm)

    # Step 2: Retrieve the answer based on the chosen RAG chain
    if "i would recommend using the code rag chain" in reasoning.lower():
        answer, retrieved_docs, _ = handle_code_query(
            user_question, code_rag_chain, llm
        )
    elif "i would recommend using the non-code rag chain" in reasoning.lower():
        answer, retrieved_docs, _ = handle_non_code_query(
            user_question, non_code_rag_chain, llm
        )
    else:
        # Default to non-code RAG chain if reasoning is unclear
        logger.warning(
            f"Unclear reasoning: {reasoning}. Defaulting to non-code RAG chain."
        )
        answer, retrieved_docs, _ = handle_non_code_query(
            user_question, non_code_rag_chain, llm
        )

    # Log the query and the LLM output to the terminal for debugging
    logger.info(f"Original Query: {user_question}")
    logger.info(f"Answer: {answer}")

    return answer, retrieved_docs, reasoning
