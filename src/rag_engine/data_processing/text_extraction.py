import os
import json
import sys
from pathlib import Path

# Add project root to path for utils import
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.logging_setup import setup_logging
from typing import List, Tuple
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter

# Set up logging
logger = setup_logging()

def initialize_semantic_chunkers() -> Tuple[CharacterTextSplitter, CharacterTextSplitter]:
    """
    Initialize semantic chunkers for natural language and code.

    Returns:
        Tuple[CharacterTextSplitter, CharacterTextSplitter]: Initialized chunkers for markdown and code.
    """
    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')  # Initialize embeddings
      # For Natural Language (Markdown)
    markdown_splitter = CharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=40
    )
    
    # For Code - Use RecursiveCharacterTextSplitter for better code splitting
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    
    code_splitter = RecursiveCharacterTextSplitter(
        chunk_size=180,
        chunk_overlap=20,
        length_function=len,
        # Use code-specific separators for better splitting
        separators=[
            "\n\ndef ",      # Function definitions
            "\n\nclass ",    # Class definitions
            "\n\n# ",        # Comments
            "\n\n",          # Double newlines
            "\ndef ",        # Function definitions (single newline)
            "\nclass ",      # Class definitions (single newline)
            "\n# ",          # Single line comments
            "\n",            # Single newlines
            " ",             # Spaces
            ""               # Characters
        ]
    )
    
    return markdown_splitter, code_splitter

def extract_text_from_files(file_paths: List[str], markdown_splitter: CharacterTextSplitter, code_splitter: CharacterTextSplitter) -> Tuple[List[str], List[str]]:
    """
    Extract text from Jupyter notebooks and Python scripts using semantic chunkers.

    Args:
        file_paths (List[str]): List of file paths to extract text from.
        markdown_splitter (CharacterTextSplitter): Chunker for markdown text.
        code_splitter (CharacterTextSplitter): Chunker for code text.

    Returns:
        Tuple[List[str], List[str]]: Extracted texts and corresponding document names.
    """
    texts = []
    doc_names = []
    for file_path in file_paths:
        try:
            if file_path.endswith('.ipynb'):
                # Extract text from Jupyter notebook cells
                with open(file_path, 'r', encoding='utf-8') as f:
                    notebook = json.load(f)
                    for cell in notebook['cells']:
                        if cell['cell_type'] == 'markdown':
                            cell_text = ' '.join(cell['source'])
                            chunks = markdown_splitter.split_text(cell_text)
                            texts.extend(chunks)
                            doc_names.extend([os.path.basename(file_path)] * len(chunks))
                        elif cell['cell_type'] == 'code':
                            cell_text = ' '.join(cell['source'])
                            chunks = code_splitter.split_text(cell_text)
                            texts.extend(chunks)
                            doc_names.extend([os.path.basename(file_path)] * len(chunks))
                logger.info(f"Extracted text from notebook: {file_path}")
            elif file_path.endswith('.py'):
                # Extract text from Python scripts
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_text = f.read()
                    chunks = code_splitter.split_text(file_text)
                    texts.extend(chunks)
                    doc_names.extend([os.path.basename(file_path)] * len(chunks))
                logger.info(f"Extracted text from script: {file_path}")
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
    return texts, doc_names