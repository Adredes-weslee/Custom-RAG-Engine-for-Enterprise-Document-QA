import os
import glob
import logging
from typing import List

def get_code_files(directory: str) -> List[str]:
    """
    Get all .py and .ipynb files from a directory.

    Args:
        directory (str): The directory to search for code files.

    Returns:
        List[str]: List of file paths.
    """
    py_files = glob.glob(os.path.join(directory, '**/*.py'), recursive=True)
    ipynb_files = glob.glob(os.path.join(directory, '**/*.ipynb'), recursive=True)
    logging.info(f"Found {len(py_files)} .py files and {len(ipynb_files)} .ipynb files.")
    return py_files + ipynb_files