import os
import json
import numpy as np
from tqdm import tqdm
from utils.logging_setup import setup_logging
from model_loader import load_models
from text_extraction import extract_text_from_files, initialize_semantic_chunkers
from file_retrieval import get_code_files
from embedding_generation import generate_code_embeddings, generate_sentence_embeddings, project_embeddings
from data_enhancement import enhance_data_with_llm
from faiss_index import create_faiss_index, save_faiss_index, load_faiss_index
from langchain_ollama.llms import OllamaLLM
from typing import List, Tuple

# Set up logging
logger = setup_logging()

# Load pre-trained models
sentence_model, code_tokenizer, code_model = load_models()
logger.info("Loaded pre-trained models.")

def main(root_directory: str) -> None:
    """
    Main function to handle data ingestion, enhancement, and embedding generation.

    Args:
        root_directory (str): Root directory containing the data.
    """
    try:
        # Initialize semantic chunkers
        markdown_splitter, code_splitter = initialize_semantic_chunkers()

        # Traverse the directory structure and get all .py and .ipynb files
        file_paths = []
        for apprentice_dir in tqdm(os.listdir(root_directory), desc="Processing apprentices"):
            apprentice_path = os.path.join(root_directory, apprentice_dir)
            if os.path.isdir(apprentice_path):
                for assignment_dir in tqdm(os.listdir(apprentice_path), desc="Processing assignments", leave=False):
                    assignment_path = os.path.join(apprentice_path, assignment_dir)
                    if os.path.isdir(assignment_path):
                        file_paths.extend(get_code_files(assignment_path))

        texts, doc_names = extract_text_from_files(file_paths, markdown_splitter, code_splitter)

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
        py_embeddings = generate_code_embeddings(py_texts, code_tokenizer, code_model)
        ipynb_embeddings = generate_sentence_embeddings(ipynb_texts, sentence_model)

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
        code_index = create_faiss_index(code_embeddings, target_dim)
        non_code_index = create_faiss_index(non_code_embeddings, target_dim)
        logger.info(f"Added {code_index.ntotal} code embeddings to the FAISS index.")
        logger.info(f"Added {non_code_index.ntotal} non-code embeddings to the FAISS index.")

        # Save the FAISS indices to disk
        save_faiss_index(code_index, 'faiss_code_index.bin')
        save_faiss_index(non_code_index, 'faiss_non_code_index.bin')

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

if __name__ == "__main__":
    # Default root directory path (adjust to your project)
    root_directory = 'data/aiap17-gitlab-data'
    main(root_directory)

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
