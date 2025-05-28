import sys
from pathlib import Path

# Add project root to path for utils import
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_ollama.llms import OllamaLLM

# ✅ Fixed import paths
from src.rag_engine.evaluation.evaluation_agent import evaluate_answer_with_ollama
from src.rag_engine.retrieval.question_handler import handle_question
from utils.logging_setup import setup_logging
from utils.model_config import get_judge_model, get_primary_model

logger = setup_logging()


def setup_streamlit_ui(
    llm: OllamaLLM = None,
    code_rag_chain: ConversationalRetrievalChain = None,
    non_code_rag_chain: ConversationalRetrievalChain = None,
) -> None:
    """
    Set up the Streamlit UI for user interaction with environment-aware model selection.

    Args:
        llm (OllamaLLM, optional): Language model instance. If None, uses primary model.
        code_rag_chain (ConversationalRetrievalChain): RAG chain instance for code-related questions.
        non_code_rag_chain (ConversationalRetrievalChain): RAG chain instance for non-code-related questions.
    """
    # Initialize primary model if not provided
    if llm is None:
        primary_model = get_primary_model()
        llm = OllamaLLM(model=primary_model)
        logger.info(f"🎯 Initialized primary model for UI: {primary_model}")

    st.title("🧠 GPU-Accelerated RAG Engine - Enterprise Document Q&A")
    st.markdown("*Environment-aware model selection with automatic GPU/CPU fallback*")

    # Show current model configuration
    with st.sidebar:
        st.subheader("🤖 Model Configuration")
        try:
            from utils.model_config import ModelConfig

            env = ModelConfig.detect_environment()
            config = ModelConfig.get_model_config(env)
            st.info(f"**Environment:** {env}")
            st.info(f"**Primary Model:** {config['primary']}")
            st.info(f"**Judge Model:** {config['judge']}")
        except Exception as e:
            st.warning(f"Model config display error: {e}")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.sidebar.subheader("Commonly Asked Questions")
    common_questions = [
        "How does this PyTorch implementation compare to best practices?",
        "Is my TensorFlow CNN implementation efficient?",
        "Are my NLP embeddings implemented correctly?",
        "Does my Flask API follow best practices?",
        "How well does my model training code work?",
    ]

    for idx, question in enumerate(common_questions):
        if st.sidebar.button(question, key=f"common_question_{idx}"):
            st.session_state.common_question = question

    # Chat interface
    st.subheader("💬 Chat History")
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if "reasoning" in message:
                with st.expander("🧠 View Reasoning"):
                    st.write(message["reasoning"])
            if message["role"] == "assistant" and "evaluation_feedback" in message:
                with st.expander("📊 Evaluation Feedback"):
                    st.write(message["evaluation_feedback"])

    # Input handling
    pre_filled_question = st.session_state.get("common_question", "")
    user_question = st.chat_input("Ask your question here...")

    if pre_filled_question and not user_question:
        user_question = pre_filled_question
        st.session_state.common_question = ""

    if user_question:
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": user_question})

        with st.chat_message("user"):
            st.write(user_question)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Processing with RAG engine..."):
                try:
                    answer, retrieved_docs, reasoning = handle_question(
                        user_question, code_rag_chain, non_code_rag_chain, llm
                    )
                    st.write(answer)

                    with st.expander("🧠 View Reasoning"):
                        st.write(reasoning)

                    if retrieved_docs:
                        with st.expander("📄 Retrieved Documents"):
                            for doc in retrieved_docs:
                                st.write(
                                    f"**Source:** {doc.metadata.get('source', 'Unknown')}"
                                )
                                st.write(f"{doc.page_content[:300]}...")

                    # Evaluate the answer
                    with st.spinner("📊 Evaluating response quality..."):
                        try:
                            judge_model = get_judge_model()
                            ollama_judge = OllamaLLM(model=judge_model)
                            evaluation_feedback = evaluate_answer_with_ollama(
                                answer, user_question, ollama_judge
                            )
                        except Exception as e:
                            logger.error(f"❌ Evaluation failed: {e}")
                            evaluation_feedback = "Evaluation temporarily unavailable."

                    # Add assistant message with evaluation
                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "reasoning": reasoning,
                            "evaluation_feedback": evaluation_feedback,
                        }
                    )

                    # Show evaluation
                    with st.expander("📊 Quality Evaluation"):
                        st.write(evaluation_feedback)

                except Exception as e:
                    st.error(f"❌ Error processing question: {e}")
                    logger.error(f"Question processing error: {e}")
