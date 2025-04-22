from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

def enhance_data_with_llm(data: str, llm: OllamaLLM) -> str:
    """
    Enhance data using an LLM to add context, improve clarity, and expand with relevant terms.

    Args:
        data (str): The original data.
        llm (OllamaLLM): The language model instance.

    Returns:
        str: The enhanced data.
    """
    template = """Rewrite the following data to make it more suitable for retrieval by adding context, improving clarity, and expanding with relevant terms:
    
    Original Data: {data}
    
    Enhanced Data:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm
    
    response = chain.invoke({"data": data})
    enhanced_data = response.strip()  # Treat response as a string
    return enhanced_data