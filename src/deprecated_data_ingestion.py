import os
import glob
import json
import logging
import faiss
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load pre-trained models
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
code_tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")
code_model = AutoModel.from_pretrained("microsoft/graphcodebert-base")
logger.info("Loaded pre-trained models.")

# Function to chunk text into smaller segments
def chunk_text(text, chunk_size=512):
    words = text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# Function to extract text from Jupyter notebooks and Python scripts
def extract_text_from_files(file_paths, chunk_size=512):
    texts = []
    doc_names = []
    for file_path in file_paths:
        try:
            if file_path.endswith('.ipynb'):
                # Extract text from Jupyter notebook cells
                with open(file_path, 'r', encoding='utf-8') as f:
                    notebook = json.load(f)
                    for cell in notebook['cells']:
                        if cell['cell_type'] == 'markdown' or cell['cell_type'] == 'code':
                            cell_text = ' '.join(cell['source'])
                            chunks = chunk_text(cell_text, chunk_size)
                            texts.extend(chunks)
                            doc_names.extend([os.path.basename(file_path)] * len(chunks))
                logger.info(f"Extracted text from notebook: {file_path}")
            elif file_path.endswith('.py'):
                # Extract text from Python files
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_text = f.read()
                    chunks = chunk_text(file_text, chunk_size)
                    texts.extend(chunks)
                    doc_names.extend([os.path.basename(file_path)] * len(chunks))
                logger.info(f"Extracted text from script: {file_path}")
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
    return texts, doc_names

# Function to get all .py and .ipynb files from a directory
def get_code_files(directory):
    py_files = glob.glob(os.path.join(directory, '**/*.py'), recursive=True)
    ipynb_files = glob.glob(os.path.join(directory, '**/*.ipynb'), recursive=True)
    logger.info(f"Found {len(py_files)} .py files and {len(ipynb_files)} .ipynb files.")
    return py_files + ipynb_files

# Function to generate embeddings for Python files using GraphCodeBERT
def generate_code_embeddings(texts):
    embeddings = []
    for text in texts:
        inputs = code_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = code_model(**inputs)
        embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy())
    return embeddings

# Function to generate embeddings for Jupyter notebooks using SentenceTransformer
def generate_sentence_embeddings(texts):
    return sentence_model.encode(texts, show_progress_bar=True)

# Function to project embeddings to a common dimensionality
def project_embeddings(embeddings, target_dim):
    projection_matrix = np.random.randn(embeddings.shape[1], target_dim)
    projected_embeddings = np.dot(embeddings, projection_matrix)
    return projected_embeddings

# Function to enhance data using an LLM
def enhance_data_with_llm(data: str, llm: OllamaLLM) -> str:
    """
    Enhance the data using an LLM to add context, improve clarity, and expand with relevant terms.

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

# Default root directory path (adjust to your project)
root_directory = 'data/aiap17-gitlab-data'

try:
    # Traverse the directory structure and get all .py and .ipynb files
    file_paths = []
    for apprentice_dir in os.listdir(root_directory):
        apprentice_path = os.path.join(root_directory, apprentice_dir)
        if os.path.isdir(apprentice_path):
            for assignment_dir in os.listdir(apprentice_path):
                assignment_path = os.path.join(apprentice_path, assignment_dir)
                if os.path.isdir(assignment_path):
                    file_paths.extend(get_code_files(assignment_path))

    texts, doc_names = extract_text_from_files(file_paths)

    if not texts:
        logger.error("No texts were extracted from the files.")
        raise ValueError("No texts were extracted.")

    # Separate texts by file type
    py_texts = [text for text, name in zip(texts, doc_names) if name.endswith('.py')]
    ipynb_texts = [text for text, name in zip(texts, doc_names) if name.endswith('.ipynb')]

    # Enhance data before generating embeddings
    llm = OllamaLLM(model="llama3.2")
    py_texts = [enhance_data_with_llm(text, llm) for text in tqdm(py_texts, desc="Enhancing Python files")]
    ipynb_texts = [enhance_data_with_llm(text, llm) for text in tqdm(ipynb_texts, desc="Enhancing Jupyter notebooks")]

    # Generate embeddings
    py_embeddings = generate_code_embeddings(py_texts)
    ipynb_embeddings = generate_sentence_embeddings(ipynb_texts)

    # Convert embeddings to numpy arrays
    py_embeddings = np.array(py_embeddings)
    ipynb_embeddings = np.array(ipynb_embeddings)

    # Project embeddings to a common dimensionality
    target_dim = 384  # Common dimensionality
    if py_embeddings.shape[1] != target_dim:
        py_embeddings = project_embeddings(py_embeddings, target_dim)
    if ipynb_embeddings.shape[1] != target_dim:
        ipynb_embeddings = project_embeddings(ipynb_embeddings, target_dim)

    # Combine embeddings and document names
    code_embeddings = py_embeddings
    non_code_embeddings = ipynb_embeddings
    code_doc_names = [name for name in doc_names if name.endswith('.py')]
    non_code_doc_names = [name for name in doc_names if name.endswith('.ipynb')]

    if len(code_embeddings) == 0 or len(non_code_embeddings) == 0:
        raise ValueError("Embeddings are empty.")

    # Print final dimensionality
    logger.info(f"Final dimensionality of code embeddings: {code_embeddings.shape[1]}")
    logger.info(f"Final dimensionality of non-code embeddings: {non_code_embeddings.shape[1]}")

    # Create FAISS indices
    code_index = faiss.IndexFlatL2(target_dim)
    non_code_index = faiss.IndexFlatL2(target_dim)
    code_index.add(code_embeddings)
    non_code_index.add(non_code_embeddings)
    logger.info(f"Added {code_index.ntotal} code embeddings to the FAISS index.")
    logger.info(f"Added {non_code_index.ntotal} non-code embeddings to the FAISS index.")

    # Save the FAISS indices to disk
    faiss.write_index(code_index, 'faiss_code_index.bin')
    faiss.write_index(non_code_index, 'faiss_non_code_index.bin')
    logger.info("FAISS indices saved to disk as 'faiss_code_index.bin' and 'faiss_non_code_index.bin'.")

    # Save document names and sources to JSON files
    code_documents = [{"content": text, "source": name} for text, name in zip(py_texts, code_doc_names)]
    non_code_documents = [{"content": text, "source": name} for text, name in zip(ipynb_texts, non_code_doc_names)]
    with open('code_docstore.json', 'w') as f:
        json.dump(code_documents, f)
    with open('non_code_docstore.json', 'w') as f:
        json.dump(non_code_documents, f)
    logger.info("Document names and sources saved to 'code_docstore.json' and 'non_code_docstore.json'.")

except Exception as e:
    logger.error(f"An error occurred during data ingestion: {e}")

# Function to load FAISS index from disk
def load_faiss_index(file_path):
    if os.path.exists(file_path):
        index = faiss.read_index(file_path)
        logger.info("FAISS index loaded from disk.")
        return index
    else:
        logger.error(f"FAISS index file {file_path} does not exist.")
        return None

# Example usage of loading the FAISS indices
code_index = load_faiss_index('faiss_code_index.bin')
non_code_index = load_faiss_index('faiss_non_code_index.bin')

if code_index is not None:
    logger.info(f"Loaded FAISS code index with {code_index.ntotal} embeddings.")
else:
    logger.error("Failed to load FAISS code index.")

if non_code_index is not None:
    logger.info(f"Loaded FAISS non-code index with {non_code_index.ntotal} embeddings.")
else:
    logger.error("Failed to load FAISS non-code index.")
    
# import os
# import glob
# import json
# import logging
# import faiss
# import torch
# import numpy as np
# from transformers import AutoTokenizer, AutoModel
# from sentence_transformers import SentenceTransformer

# # Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Load pre-trained models
# sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
# code_tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")
# code_model = AutoModel.from_pretrained("microsoft/graphcodebert-base")
# logger.info("Loaded pre-trained models.")

# # Function to chunk text into smaller segments
# def chunk_text(text, chunk_size=512):
#     words = text.split()
#     return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# # Function to extract text from Jupyter notebooks and Python scripts
# def extract_text_from_files(file_paths, chunk_size=512):
#     texts = []
#     doc_names = []
#     for file_path in file_paths:
#         try:
#             if file_path.endswith('.ipynb'):
#                 # Extract text from Jupyter notebook cells
#                 with open(file_path, 'r', encoding='utf-8') as f:
#                     notebook = json.load(f)
#                     for cell in notebook['cells']:
#                         if cell['cell_type'] == 'markdown' or cell['cell_type'] == 'code':
#                             cell_text = ' '.join(cell['source'])
#                             chunks = chunk_text(cell_text, chunk_size)
#                             texts.extend(chunks)
#                             doc_names.extend([os.path.basename(file_path)] * len(chunks))
#                 logger.info(f"Extracted text from notebook: {file_path}")
#             elif file_path.endswith('.py'):
#                 # Extract text from Python files
#                 with open(file_path, 'r', encoding='utf-8') as f:
#                     file_text = f.read()
#                     chunks = chunk_text(file_text, chunk_size)
#                     texts.extend(chunks)
#                     doc_names.extend([os.path.basename(file_path)] * len(chunks))
#                 logger.info(f"Extracted text from script: {file_path}")
#         except Exception as e:
#             logger.error(f"Error processing file {file_path}: {e}")
#     return texts, doc_names

# # Function to get all .py and .ipynb files from a directory
# def get_code_files(directory):
#     py_files = glob.glob(os.path.join(directory, '**/*.py'), recursive=True)
#     ipynb_files = glob.glob(os.path.join(directory, '**/*.ipynb'), recursive=True)
#     logger.info(f"Found {len(py_files)} .py files and {len(ipynb_files)} .ipynb files.")
#     return py_files + ipynb_files

# # Function to generate embeddings for Python files using GraphCodeBERT
# def generate_code_embeddings(texts):
#     embeddings = []
#     for text in texts:
#         inputs = code_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
#         with torch.no_grad():
#             outputs = code_model(**inputs)
#         embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy())
#     return embeddings

# # Function to generate embeddings for Jupyter notebooks using SentenceTransformer
# def generate_sentence_embeddings(texts):
#     return sentence_model.encode(texts, show_progress_bar=True)

# # Function to project embeddings to a common dimensionality
# def project_embeddings(embeddings, target_dim):
#     projection_matrix = np.random.randn(embeddings.shape[1], target_dim)
#     projected_embeddings = np.dot(embeddings, projection_matrix)
#     return projected_embeddings

# # Default root directory path (adjust to your project)
# root_directory = 'data/aiap17-gitlab-data'

# try:
#     # Traverse the directory structure and get all .py and .ipynb files
#     file_paths = []
#     for apprentice_dir in os.listdir(root_directory):
#         apprentice_path = os.path.join(root_directory, apprentice_dir)
#         if os.path.isdir(apprentice_path):
#             for assignment_dir in os.listdir(apprentice_path):
#                 assignment_path = os.path.join(apprentice_path, assignment_dir)
#                 if os.path.isdir(assignment_path):
#                     file_paths.extend(get_code_files(assignment_path))

#     texts, doc_names = extract_text_from_files(file_paths)

#     if not texts:
#         logger.error("No texts were extracted from the files.")
#         raise ValueError("No texts were extracted.")

#     # Separate texts by file type
#     py_texts = [text for text, name in zip(texts, doc_names) if name.endswith('.py')]
#     ipynb_texts = [text for text, name in zip(texts, doc_names) if name.endswith('.ipynb')]

#     # Generate embeddings
#     py_embeddings = generate_code_embeddings(py_texts)
#     ipynb_embeddings = generate_sentence_embeddings(ipynb_texts)

#     # Convert embeddings to numpy arrays
#     py_embeddings = np.array(py_embeddings)
#     ipynb_embeddings = np.array(ipynb_embeddings)

#     # Project embeddings to a common dimensionality
#     target_dim = 384  # Common dimensionality
#     if py_embeddings.shape[1] != target_dim:
#         py_embeddings = project_embeddings(py_embeddings, target_dim)
#     if ipynb_embeddings.shape[1] != target_dim:
#         ipynb_embeddings = project_embeddings(ipynb_embeddings, target_dim)

#     # Combine embeddings and document names
#     embeddings = np.vstack((py_embeddings, ipynb_embeddings))
#     doc_names = [name for name in doc_names if name.endswith('.py')] + [name for name in doc_names if name.endswith('.ipynb')]

#     if len(embeddings) == 0:
#         raise ValueError("Embeddings are empty.")

#     # Print final dimensionality
#     logger.info(f"Final dimensionality of embeddings: {embeddings.shape[1]}")

#     # Create FAISS index
#     index = faiss.IndexFlatL2(target_dim)
#     index.add(embeddings)
#     logger.info(f"Added {index.ntotal} embeddings to the FAISS index.")

#     # Save the FAISS index to disk
#     faiss.write_index(index, 'faiss_index.bin')
#     logger.info("FAISS index saved to disk as 'faiss_index.bin'.")

#     # Save document names and sources to a JSON file
#     documents = [{"content": text, "source": name} for text, name in zip(texts, doc_names)]
#     with open('docstore.json', 'w') as f:
#         json.dump(documents, f)
#     logger.info("Document names and sources saved to 'docstore.json'.")

# except Exception as e:
#     logger.error(f"An error occurred during data ingestion: {e}")

# # Function to load FAISS index from disk
# def load_faiss_index(file_path):
#     if os.path.exists(file_path):
#         index = faiss.read_index(file_path)
#         logger.info("FAISS index loaded from disk.")
#         return index
#     else:
#         logger.error(f"FAISS index file {file_path} does not exist.")
#         return None

# # Example usage of loading the FAISS index
# index = load_faiss_index('faiss_index.bin')

# if index is not None:
#     logger.info(f"Loaded FAISS index with {index.ntotal} embeddings.")
# else:
#     logger.error("Failed to load FAISS index.")