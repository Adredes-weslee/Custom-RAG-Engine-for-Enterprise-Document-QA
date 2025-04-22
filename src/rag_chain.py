from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain.chat_models import AzureChatOpenAI

def setup_rag_chain(llm: AzureChatOpenAI, vector_store: FAISS, top_k: int) -> ConversationalRetrievalChain:
    """
    Set up the RAG chain for conversational retrieval.

    Args:
        llm (AzureChatOpenAI): Language model instance.
        vector_store (FAISS): FAISS vector store instance.
        top_k (int): Number of top results to retrieve.

    Returns:
        ConversationalRetrievalChain: Configured RAG chain.
    """
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
    rag_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True
    )
    return rag_chain