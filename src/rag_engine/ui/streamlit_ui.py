import streamlit as st
from question_handler import handle_question
from langchain_ollama.llms import OllamaLLM
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
# from evaluation_agent import evaluate_answer_with_azure
from evaluation_agent import evaluate_answer_with_ollama
from langchain.chat_models import AzureChatOpenAI
from config.config import endpoint, api_key, model, api_version, model_version

def setup_streamlit_ui(llm: OllamaLLM, code_rag_chain: ConversationalRetrievalChain, non_code_rag_chain: ConversationalRetrievalChain) -> None:
    """
    Set up the Streamlit UI for user interaction.

    Args:
        llm (OllamaLLM): Language model instance.
        code_rag_chain (ConversationalRetrievalChain): RAG chain instance for code-related questions.
        non_code_rag_chain (ConversationalRetrievalChain): RAG chain instance for non-code-related questions.
    """
    st.title("AIAP Batch 17 Repository Q&A Platform")

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    st.sidebar.subheader("Commonly Asked Questions")
    common_questions = [
        "How does this PyTorch implementation of sentiment analysis compare to best practices? (Please provide your code)",
        "Is my TensorFlow CNN implementation efficient compared to similar models? (Please include your code for evaluation)",
        "Are the contextual embeddings in my NLP model implemented correctly? (Submit your code for evaluation)",
        "Does my Flask REST API setup follow best coding practices? (Please provide your code)",
        "Can you evaluate the correctness of my supervised vs. unsupervised learning implementation? (Include your code for comparison)",
        "How well does my fine-tuned BERT model adhere to best practices? (Submit your code for evaluation)",
        "Is my transfer learning setup optimal? (Please include your code for evaluation)",
        "How does my approach to deploying a machine learning model using Docker compare to best practices? (Submit your code for review)",
        "Is my GAN implementation correct, and does it follow best practices? (Please provide your code)",
        "How does my handling of imbalanced datasets compare to best practices? (Submit your code for evaluation)"
    ]

    for idx, question in enumerate(common_questions):
        if st.sidebar.button(question, key=f"common_question_{idx}"):
            st.session_state.common_question = question

    st.sidebar.subheader("Last Query")
    if st.session_state.chat_history:
        # Find the last user query
        last_query = next((msg["content"] for msg in reversed(st.session_state.chat_history) if msg["role"] == "user"), None)
        if last_query and st.sidebar.button(last_query, key="last_query"):
            st.session_state.common_question = last_query

    st.subheader("Chat History")
    for i, message in enumerate(st.session_state.chat_history):
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if "reasoning" in message:
                with st.expander("View Reasoning and Code"):
                    st.write(message["reasoning"])
            if message["role"] == "assistant" and "evaluation_feedback" in message:
                st.write(f"**Evaluation Feedback:** {message['evaluation_feedback']}")

    pre_filled_question = st.session_state.get("common_question", "")
    user_question = st.chat_input("Ask your question here:")

    if pre_filled_question and not user_question:
        user_question = pre_filled_question
        st.session_state.common_question = ""

    if user_question:
        st.session_state.chat_input = ""
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.write(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving answer..."):
                answer, retrieved_docs, reasoning = handle_question(user_question, code_rag_chain, non_code_rag_chain, llm)
                st.write(answer)
                with st.expander("View Reasoning"):
                    st.write(reasoning)
                if retrieved_docs:
                    with st.expander("Retrieved Documents"):
                        for doc in retrieved_docs:
                            st.write(f"**Document Source:** {doc.metadata.get('source', 'Unknown')}")
                            st.write(f"{doc.page_content[:500]}")
        
        # Evaluate the answer using Ollama Gemma2
        ollama_llm = OllamaLLM(model="gemma2")
        evaluation_feedback = evaluate_answer_with_ollama(answer, user_question, ollama_llm)
                
        ##### Uncomment if using AzureChatOpenAI model instead of OllamaLLM model. #####
        
        # # Evaluate the answer using Azure OpenAI
        # azure_llm = AzureChatOpenAI(
        #     azure_endpoint=endpoint,
        #     openai_api_key=api_key,
        #     model_name=model,
        #     openai_api_version=api_version,
        #     model_version=model_version
        # )
        # evaluation_feedback = evaluate_answer_with_azure(answer, user_question, azure_llm)
        
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "reasoning": reasoning,
            "evaluation_feedback": evaluation_feedback
        })

        st.subheader("Evaluation Feedback")
        st.write(evaluation_feedback)
        
##### The following code snippet is for using the AzureChatOpenAI model instead of the OllamaLLM model. #####


# import streamlit as st
# from question_handler import handle_question
# from langchain.chat_models import AzureChatOpenAI
# from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain

# def setup_streamlit_ui(llm: AzureChatOpenAI, code_rag_chain: ConversationalRetrievalChain, non_code_rag_chain: ConversationalRetrievalChain) -> None:
#     """
#     Set up the Streamlit UI for user interaction.

#     Args:
#         llm (AzureChatOpenAI): Language model instance.
#         code_rag_chain (ConversationalRetrievalChain): RAG chain instance for code-related questions.
#         non_code_rag_chain (ConversationalRetrievalChain): RAG chain instance for non-code-related questions.
#     """
#     st.title("AIAP Batch Repository Q&A Platform")

#     if 'chat_history' not in st.session_state:
#         st.session_state.chat_history = []

#     st.sidebar.subheader("Commonly Asked Questions")
#     common_questions = [
#         "How do I perform sentiment analysis using PyTorch?",
#         "Can you provide an example of a CNN implementation in TensorFlow?",
#         "What are contextual embeddings and how are they used in NLP?",
#         "How do I set up a REST API with Flask?",
#         "What is the difference between supervised and unsupervised learning?"
#     ]

#     for question in common_questions:
#         if st.sidebar.button(question):
#             st.session_state.common_question = question

#     st.subheader("Chat History")
#     for message in st.session_state.chat_history:
#         with st.chat_message(message["role"]):
#             st.write(message["content"])
#             if "reasoning" in message:
#                 with st.expander("View Reasoning and Code"):
#                     st.write(message["reasoning"])

#     pre_filled_question = st.session_state.get("common_question", "")
#     user_question = st.chat_input("Ask your question here:")

#     if pre_filled_question and not user_question:
#         user_question = pre_filled_question
#         st.session_state.common_question = ""

#     if user_question:
#         st.session_state.chat_input = ""
#         st.session_state.chat_history.append({"role": "user", "content": user_question})
#         with st.chat_message("user"):
#             st.write(user_question)

#         with st.chat_message("assistant"):
#             with st.spinner("Retrieving answer..."):
#                 answer, retrieved_docs, reasoning = handle_question(user_question, code_rag_chain, non_code_rag_chain, llm)
#                 st.write(answer)
#                 with st.expander("View Reasoning"):
#                     st.write(reasoning)
#                 if retrieved_docs:
#                     with st.expander("Retrieved Documents"):
#                         for doc in retrieved_docs:
#                             st.write(f"**Document Source:** {doc.metadata.get('source', 'Unknown')}")
#                             st.write(f"{doc.page_content[:500]}")
        
#         st.session_state.chat_history.append({"role": "assistant", "content": answer, "reasoning": reasoning})