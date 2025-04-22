import os
import json
import logging
from typing import List, Tuple
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter


def initialize_semantic_chunkers() -> Tuple[CharacterTextSplitter, CharacterTextSplitter]:
    """
    Initialize semantic chunkers for natural language and code.

    Returns:
        Tuple[CharacterTextSplitter, CharacterTextSplitter]: Initialized chunkers for markdown and code.
    """
    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')  # Initialize embeddings

    # For Natural Language (Markdown)
    markdown_splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    
    # For Code
    code_splitter = CharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
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
                logging.info(f"Extracted text from notebook: {file_path}")
            elif file_path.endswith('.py'):
                # Extract text from Python scripts
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_text = f.read()
                    chunks = code_splitter.split_text(file_text)
                    texts.extend(chunks)
                    doc_names.extend([os.path.basename(file_path)] * len(chunks))
                logging.info(f"Extracted text from script: {file_path}")
        except Exception as e:
            logging.error(f"Error processing file {file_path}: {e}")
    return texts, doc_names


##### Uncomment to use vanilla chunking method #####

# import os
# import json
# import logging
# from typing import List, Tuple

# def chunk_text(text: str, chunk_size: int = 512) -> List[str]:
#     """
#     Chunk text into smaller segments.

#     Args:
#         text (str): The text to be chunked.
#         chunk_size (int, optional): The size of each chunk. Defaults to 512.

#     Returns:
#         List[str]: List of text chunks.
#     """
#     words = text.split()
#     return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# def extract_text_from_files(file_paths: List[str], chunk_size: int = 512) -> Tuple[List[str], List[str]]:
#     """
#     Extract text from Jupyter notebooks and Python scripts.

#     Args:
#         file_paths (List[str]): List of file paths to extract text from.
#         chunk_size (int, optional): The size of each chunk. Defaults to 512.

#     Returns:
#         Tuple[List[str], List[str]]: Extracted texts and corresponding document names.
#     """
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
#                 logging.info(f"Extracted text from notebook: {file_path}")
#             elif file_path.endswith('.py'):
#                 # Extract text from Python scripts
#                 with open(file_path, 'r', encoding='utf-8') as f:
#                     file_text = f.read()
#                     chunks = chunk_text(file_text, chunk_size)
#                     texts.extend(chunks)
#                     doc_names.extend([os.path.basename(file_path)] * len(chunks))
#                 logging.info(f"Extracted text from script: {file_path}")
#         except Exception as e:
#             logging.error(f"Error processing file {file_path}: {e}")
#     return texts, doc_names