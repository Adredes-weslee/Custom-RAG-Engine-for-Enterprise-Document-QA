import json
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from langchain_ollama.llms import OllamaLLM

from rag_engine.data_processing.data_enhancement import enhance_data_with_llm
from rag_engine.data_processing.file_retrieval import get_code_files
from rag_engine.data_processing.text_extraction import (
    extract_text_from_files,
    initialize_semantic_chunkers,
)
from rag_engine.embeddings.embedding_generation import (
    generate_code_embeddings,
    generate_sentence_embeddings,
    project_embeddings,
)
from rag_engine.embeddings.faiss_index import (
    create_faiss_index,
    load_faiss_index,
    save_faiss_index,
)
from rag_engine.embeddings.model_loader import load_models
from utils.logging_setup import setup_logging
from utils.model_config import ModelConfig, get_primary_model  # ← ADD THIS

# Set up logging
logger = setup_logging()


# ✅ Environment-aware model loading
def get_data_enhancement_llm() -> OllamaLLM:
    """Get LLM for data enhancement based on environment."""
    primary_model = get_primary_model()  # ← Uses your hybrid system!
    logger.info(f"🤖 Using model for data enhancement: {primary_model}")

    try:
        return OllamaLLM(model=primary_model)
    except Exception as e:
        logger.warning(f"⚠️ Failed to load primary model {primary_model}: {e}")
        # Fallback to smallest model
        from utils.model_config import get_fallback_model

        fallback_model = get_fallback_model()
        logger.info(f"🔄 Falling back to: {fallback_model}")
        return OllamaLLM(model=fallback_model)


# Load pre-trained models
sentence_model, code_tokenizer, code_model = load_models()
logger.info("Loaded pre-trained models.")


def main(
    root_directory: str, limit_people: int = None, limit_files_per_person: int = None
) -> None:
    """
    Main function to handle data ingestion, enhancement, and embedding generation.
    Now with environment-aware model selection and optional limits for testing.

    Args:
        root_directory (str): Root directory containing the data.
        limit_people (int, optional): Limit number of people to process (for testing)
        limit_files_per_person (int, optional): Limit files per person (for testing)
    """
    try:
        # ✅ Show environment info
        env = ModelConfig.detect_environment()
        config = ModelConfig.get_model_config(env)
        logger.info(f"🎯 Environment: {env}")
        logger.info(f"📋 Model config: {config}")

        # Initialize semantic chunkers
        markdown_splitter, code_splitter = initialize_semantic_chunkers()

        # Traverse the directory structure and get all .py and .ipynb files
        file_paths = []
        people_dirs = [
            d
            for d in os.listdir(root_directory)
            if os.path.isdir(os.path.join(root_directory, d))
        ]

        # ✅ Optional limiting for testing
        if limit_people:
            people_dirs = people_dirs[:limit_people]
            logger.info(f"🧪 Testing mode: Processing only {len(people_dirs)} people")

        for apprentice_dir in tqdm(people_dirs, desc="Processing apprentices"):
            apprentice_path = os.path.join(root_directory, apprentice_dir)
            if os.path.isdir(apprentice_path):
                person_files = []
                for assignment_dir in tqdm(
                    os.listdir(apprentice_path),
                    desc="Processing assignments",
                    leave=False,
                ):
                    assignment_path = os.path.join(apprentice_path, assignment_dir)
                    if os.path.isdir(assignment_path):
                        person_files.extend(get_code_files(assignment_path))

                # ✅ Optional file limiting per person
                if limit_files_per_person:
                    person_files = person_files[:limit_files_per_person]

                file_paths.extend(person_files)

        logger.info(f"📄 Found {len(file_paths)} total files to process")

        if not file_paths:
            logger.error("❌ No files found to process!")
            return

        texts, doc_names = extract_text_from_files(
            file_paths, markdown_splitter, code_splitter
        )

        if not texts:
            logger.error("No texts were extracted from the files.")
            raise ValueError("No texts were extracted.")

        # Separate texts by file type
        py_texts = [
            text for text, name in zip(texts, doc_names) if name.endswith(".py")
        ]
        ipynb_texts = [
            text for text, name in zip(texts, doc_names) if name.endswith(".ipynb")
        ]

        logger.info(
            f"📊 Python files: {len(py_texts)}, Jupyter notebooks: {len(ipynb_texts)}"
        )

        # ✅ Environment-aware LLM for enhancement
        llm = get_data_enhancement_llm()

        # Enhance data before generating embeddings
        if py_texts:
            logger.info("🔧 Enhancing Python files...")
            py_texts = [
                enhance_data_with_llm(text, llm)
                for text in tqdm(py_texts, desc="Enhancing Python files")
            ]

        if ipynb_texts:
            logger.info("🔧 Enhancing Jupyter notebooks...")
            ipynb_texts = [
                enhance_data_with_llm(text, llm)
                for text in tqdm(ipynb_texts, desc="Enhancing Jupyter notebooks")
            ]

        # Generate embeddings
        py_embeddings = (
            generate_code_embeddings(py_texts, code_tokenizer, code_model)
            if py_texts
            else np.array([])
        )
        ipynb_embeddings = (
            generate_sentence_embeddings(ipynb_texts, sentence_model)
            if ipynb_texts
            else np.array([])
        )

        # Convert embeddings to numpy arrays
        if len(py_embeddings) > 0:
            py_embeddings = np.array(py_embeddings)
        if len(ipynb_embeddings) > 0:
            ipynb_embeddings = np.array(ipynb_embeddings)

        # Project embeddings to a common dimensionality
        target_dim = 384  # Common dimensionality
        if len(py_embeddings) > 0 and py_embeddings.shape[1] != target_dim:
            py_embeddings = project_embeddings(py_embeddings, target_dim)
        if len(ipynb_embeddings) > 0 and ipynb_embeddings.shape[1] != target_dim:
            ipynb_embeddings = project_embeddings(ipynb_embeddings, target_dim)

        # Combine embeddings and document names
        code_embeddings = py_embeddings
        non_code_embeddings = ipynb_embeddings
        code_doc_names = [name for name in doc_names if name.endswith(".py")]
        non_code_doc_names = [name for name in doc_names if name.endswith(".ipynb")]

        if len(code_embeddings) == 0 and len(non_code_embeddings) == 0:
            raise ValueError("No embeddings were generated.")

        # Print final dimensionality
        if len(code_embeddings) > 0:
            logger.info(
                f"Final dimensionality of code embeddings: {code_embeddings.shape[1]}"
            )
        if len(non_code_embeddings) > 0:
            logger.info(
                f"Final dimensionality of non-code embeddings: {non_code_embeddings.shape[1]}"
            )

        # Create FAISS indices
        if len(code_embeddings) > 0:
            code_index = create_faiss_index(code_embeddings, target_dim)
            logger.info(
                f"Added {code_index.ntotal} code embeddings to the FAISS index."
            )
            save_faiss_index(code_index, "faiss_code_index.bin")

            # Save code documents
            code_documents = [
                {"content": text, "source": name}
                for text, name in zip(py_texts, code_doc_names)
            ]
            with open("code_docstore.json", "w") as f:
                json.dump(code_documents, f)
            logger.info("✅ Code index and docstore saved")

        if len(non_code_embeddings) > 0:
            non_code_index = create_faiss_index(non_code_embeddings, target_dim)
            logger.info(
                f"Added {non_code_index.ntotal} non-code embeddings to the FAISS index."
            )
            save_faiss_index(non_code_index, "faiss_non_code_index.bin")

            # Save non-code documents
            non_code_documents = [
                {"content": text, "source": name}
                for text, name in zip(ipynb_texts, non_code_doc_names)
            ]
            with open("non_code_docstore.json", "w") as f:
                json.dump(non_code_documents, f)
            logger.info("✅ Non-code index and docstore saved")

        logger.info("🎉 Data ingestion completed successfully!")

    except Exception as e:
        logger.error(f"An error occurred during data ingestion: {e}")
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Data ingestion with environment-aware model selection"
    )
    parser.add_argument(
        "--root-directory",
        default="data/aiap17-gitlab-data",
        help="Root directory containing data",
    )
    parser.add_argument(
        "--limit-people",
        type=int,
        help="Limit number of people to process (for testing)",
    )
    parser.add_argument(
        "--limit-files", type=int, help="Limit files per person (for testing)"
    )
    parser.add_argument(
        "--test", action="store_true", help="Run in test mode (3 people, 10 files each)"
    )

    args = parser.parse_args()

    if args.test:
        logger.info("🧪 Running in test mode")
        main(args.root_directory, limit_people=3, limit_files_per_person=10)
    else:
        main(args.root_directory, args.limit_people, args.limit_files)

    # Example usage of loading the FAISS indices
    try:
        code_index = load_faiss_index("faiss_code_index.bin")
        non_code_index = load_faiss_index("faiss_non_code_index.bin")

        if code_index is not None:
            logger.info(f"Loaded FAISS code index with {code_index.ntotal} embeddings.")
        if non_code_index is not None:
            logger.info(
                f"Loaded FAISS non-code index with {non_code_index.ntotal} embeddings."
            )
    except Exception as e:
        logger.warning(f"Could not verify index loading: {e}")
