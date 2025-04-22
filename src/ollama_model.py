from langchain_ollama.llms import OllamaLLM

def load_ollama_model() -> OllamaLLM:
    """
    Load and initialize the Ollama model.
    
    Returns:
        OllamaLLM: Initialized language model instance.
    """
    # Initialize the Ollama model
    return OllamaLLM(model="llama3.2")

def ollama_llm() -> OllamaLLM:
    """
    Return the Ollama LLM instance.
    
    Returns:
        OllamaLLM: Language model instance.
    """
    return load_ollama_model()


##### The below code is for using the AzureChatOpenAI model instead of the OllamaLLM model. #####

# from langchain.chat_models import AzureChatOpenAI
# from config import endpoint, api_key, model, api_version, model_version

# def load_ollama_model() -> AzureChatOpenAI:
#     """
#     Load and initialize the Ollama model.
    
#     Returns:
#         AzureChatOpenAI: Initialized language model instance.
#     """
#     # Replace this with actual code to load and initialize the Ollama model
#     # For now, we will return a mock AzureChatOpenAI instance for testing purposes
#     return AzureChatOpenAI(
#         azure_endpoint=endpoint,
#         openai_api_key=api_key,
#         model_name=model,
#         openai_api_version=api_version,
#         model_version=model_version,
#         temperature=0.7,
#         max_tokens=4096,
#         model_kwargs={"top_p": 0.5}
#     )

# def ollama_llm() -> AzureChatOpenAI:
#     """
#     Return the Ollama LLM instance.
    
#     Returns:
#         AzureChatOpenAI: Language model instance.
#     """
#     return load_ollama_model()