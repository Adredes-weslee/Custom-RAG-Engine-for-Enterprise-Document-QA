import os
import json
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.chat_models import AzureChatOpenAI
from langchain.schema import HumanMessage
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.callbacks.streamlit import StreamlitCallbackHandler
from langchain.docstore import InMemoryDocstore
from langchain.docstore.document import Document
import faiss
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Fetch credentials from environment variables
endpoint = os.getenv("ENDPOINT")
api_key = os.getenv("API_KEY")
model = os.getenv("MODEL")
api_version = os.getenv("API_VERSION")
model_version = os.getenv("MODEL_VERSION")

# Initialize Azure OpenAI LLM
llm = AzureChatOpenAI(
    azure_endpoint=endpoint,
    openai_api_key=api_key,
    model_name=model,
    openai_api_version=api_version,
    model_version=model_version,
    temperature=0.7,
    max_tokens=300,
    model_kwargs={"top_p": 0.5}
)

# Function to load FAISS index from disk
def load_faiss_index(file_path):
    if os.path.exists(file_path):
        logger.info(f"FAISS index file found at {file_path}.")
        index = faiss.read_index(file_path)
        logger.info("FAISS index loaded from disk.")
        return index
    else:
        logger.error(f"FAISS index file {file_path} does not exist.")
        return None

# Load the FAISS index from disk
faiss_index_path = './faiss_index.bin'
faiss_index = load_faiss_index(faiss_index_path)

# If FAISS index is not found, log an error and stop execution
if faiss_index is None:
    st.error("FAISS index not found. Please ensure the FAISS index file exists at the specified path.")
    st.stop()

# Load documents with metadata from JSON file
docstore_path = './docstore.json'
if os.path.exists(docstore_path):
    with open(docstore_path, 'r') as f:
        documents = json.load(f)
    logger.info("Documents loaded from 'docstore.json'.")
else:
    st.error("Document store file not found. Please ensure 'docstore.json' exists.")
    st.stop()

# Create a document store and index to document store ID mapping
docstore = InMemoryDocstore({i: Document(page_content=doc["content"], metadata={"source": doc["source"]}) for i, doc in enumerate(documents)})
index_to_docstore_id = {i: i for i in range(len(documents))}

# Ensure the index_to_docstore_id mapping covers all indices in the FAISS index
num_vectors = faiss_index.ntotal
if num_vectors != len(index_to_docstore_id):
    st.warning(f"Mismatch between number of vectors in FAISS index ({num_vectors}) and document store IDs ({len(index_to_docstore_id)}). Proceeding with available data.")
    index_to_docstore_id = {i: i % len(documents) for i in range(num_vectors)}

# Wrap the FAISS index in a LangChain-compatible vector store
model = SentenceTransformer('all-MiniLM-L6-v2')
vector_store = FAISS(
    embedding_function=model.encode,
    index=faiss_index,
    docstore=docstore,
    index_to_docstore_id=index_to_docstore_id
)

# Function to set up retrieval and question answering (RAG)
def setup_rag_chain(llm, vector_store, top_k):
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
    rag_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True
    )
    return rag_chain

# Function to create a Pandas Agent for CSV analysis with clear dataset context
def setup_pandas_agent_with_context(llm, dataframe):
    system_prompt = f"""
    You are working with a dataset containing information about various code repositories. The dataset has the following columns:
    {', '.join(dataframe.columns)}

    Important: This dataset contains information from multiple repositories. You do not need to look for a separate dataset for each repository.

    When answering questions:
    1. Always consider the entire dataset, which includes data from all repositories.
    2. Use the 'Repository_Name' column to distinguish between the repositories when necessary.
    3. For questions about specific code snippets or functions, look for relevant information in the 'Code_Snippet' and 'Function_Name' columns.

    Please perform analysis dynamically based on the columns provided and the instructions above.

    Additionally, you are capable of retrieving code snippets and providing explanations or context for the code. When a user asks a question about how to perform a specific task or requests a code example, retrieve the relevant code snippet and provide a clear explanation.

    Example questions you might encounter:
    - "How do I perform sentiment analysis using PyTorch?"
    - "Can you provide an example of a CNN implementation in TensorFlow?"
    - "What are contextual embeddings and how are they used in NLP?"

    Please ensure your responses are accurate and provide the necessary context or code examples to help the user understand and implement the solution.
    """
    
    pandas_agent = create_pandas_dataframe_agent(
        llm,
        dataframe,
        system_prompt=system_prompt,
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        handle_parsing_errors=True,
        allow_dangerous_code=True
    )
    return pandas_agent

# Function to let LLM determine whether to use Pandas Agent or RAG for a question
def determine_approach(user_question, pandas_agent, rag_chain):
    reasoning_prompt = f"""
    You are an AI assistant. The provided dataset contains information about various code repositories, with the 'Repository_Name' column indicating the repository.
    
    Your task is to decide whether the user's question should be answered using RAG (for code retrieval from the repository) or Pandas Agent (for CSV analysis).
    
    If the question is about code snippets, functions, or specific implementation details, RAG should be used.
    If the question is about data analysis, statistics, or specific repository metadata, Pandas Agent should be used.

    Here is the user's question: "{user_question}"

    Which approach should be used: Pandas Agent or RAG? Provide reasoning.
    """
    
    response = llm([HumanMessage(content=reasoning_prompt)])
    reasoning = response.content
    return reasoning

# Function to handle the user's question using StreamlitCallbackHandler for logging
def handle_question(user_question, pandas_agent, rag_chain):
    reasoning = determine_approach(user_question, pandas_agent, rag_chain)

    if "pandas agent" in reasoning.lower():
        full_question = f"""
        Remember, you are working with a dataset containing information about various code repositories.
        The 'Repository_Name' column indicates which repository each row belongs to.
        
        User question: {user_question}
        
        Please analyze the data with the appropriate functions and provide a clear, concise answer.
        """

        log_container = st.container()
        streamlit_callback = StreamlitCallbackHandler(parent_container=log_container)
        result = pandas_agent.run(full_question, callbacks=[streamlit_callback])
        return result, [], reasoning
    else:
        response = rag_chain({"question": user_question})
        return response['answer'], response["source_documents"], reasoning
    
# Streamlit UI elements
st.title("AIAP Batch Repository Q&A Platform")

# Initialize session state for chat history if it doesn't exist
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Setup RAG chain
top_k = 25  # You can make this configurable via Streamlit UI if needed
rag_chain = setup_rag_chain(llm, vector_store, top_k)

# Setup Pandas Agent (assuming you have a pre-loaded dataframe)
# For demonstration, we'll create an empty dataframe
merged_csv_df = pd.DataFrame(columns=["Repository_Name", "Code_Snippet", "Function_Name"])
pandas_agent = setup_pandas_agent_with_context(llm, merged_csv_df)

# Function to handle commonly asked questions
def set_common_question(question):
    st.session_state.common_question = question

# Commonly Asked Questions in the Sidebar
st.sidebar.subheader("Commonly Asked Questions")
common_questions = [
    "How do I perform sentiment analysis using PyTorch?",
    "Can you provide an example of a CNN implementation in TensorFlow?",
    "What are contextual embeddings and how are they used in NLP?",
    "How do I set up a REST API with Flask?",
    "What is the difference between supervised and unsupervised learning?"
]

# Sidebar buttons for commonly asked questions
for question in common_questions:
    if st.sidebar.button(question):
        set_common_question(question)

# Display chat history
st.subheader("Chat History")
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if "reasoning" in message:
            with st.expander("View Reasoning and Code"):
                st.write(message["reasoning"])

# Get the pre-filled question from the session state
pre_filled_question = st.session_state.get("common_question", "")

# User input
user_question = st.chat_input("Ask your question here:")

# If there's a pre-filled question from common questions, set the input text
if pre_filled_question and not user_question:
    user_question = pre_filled_question
    st.session_state.common_question = ""  # Clear after use

if user_question:
    st.session_state.chat_input = ""  # Clear input box after use
    st.session_state.chat_history.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.write(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving answer..."):
            answer, retrieved_docs, reasoning = handle_question(user_question, pandas_agent, rag_chain)
            st.write(answer)
            with st.expander("View Reasoning"):
                st.write(reasoning)
            if retrieved_docs:
                with st.expander("Retrieved Documents"):
                    for doc in retrieved_docs:
                        st.write(f"**Document Source:** {doc.metadata.get('source', 'Unknown')}")
                        st.write(f"{doc.page_content[:500]}")
    
    st.session_state.chat_history.append({"role": "assistant", "content": answer, "reasoning": reasoning})