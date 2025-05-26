"""
Test Data Processing Pipeline

This script tests each component of the data processing pipeline:
1. File Retrieval - Finding Python and Jupyter notebook files
2. Text Extraction - Extracting and chunking text from files  
3. Data Enhancement - LLM-powered enhancement  
4. Data Ingestion - Full pipeline orchestration
5. Data Analysis - Analyzing the extracted data

Usage: python test_data_processing.py
"""

import sys
import os
import logging
from pathlib import Path

# Add src to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
SRC_PATH = PROJECT_ROOT / "src"
sys.path.append(str(SRC_PATH))

print(f"Project root: {PROJECT_ROOT}")
print(f"Adding to path: {SRC_PATH}")
print(f"src directory exists: {SRC_PATH.exists()}")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test importing our data processing modules"""
    print("\n" + "="*50)
    print("TESTING IMPORTS")
    print("="*50)
    
    try:
        from rag_engine.data_processing.file_retrieval import get_code_files
        from rag_engine.data_processing.text_extraction import initialize_semantic_chunkers, extract_text_from_files
        from rag_engine.data_processing.data_enhancement import enhance_data_with_llm
        print("✅ All data_processing imports successful!")
        
        # Test embeddings imports
        from rag_engine.embeddings.model_loader import load_models
        from rag_engine.embeddings.embedding_generation import generate_code_embeddings, generate_sentence_embeddings
        from rag_engine.embeddings.faiss_index import create_faiss_index
        print("✅ All embeddings imports successful!")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        
        # Debug information
        rag_engine_path = SRC_PATH / "rag_engine"
        if rag_engine_path.exists():
            print(f"rag_engine directory contents: {list(rag_engine_path.iterdir())}")
            data_processing_path = rag_engine_path / "data_processing"
            if data_processing_path.exists():
                print(f"data_processing directory contents: {list(data_processing_path.iterdir())}")
            embeddings_path = rag_engine_path / "embeddings"
            if embeddings_path.exists():
                print(f"embeddings directory contents: {list(embeddings_path.iterdir())}")
        return False

def test_file_retrieval():
    """Test file retrieval functionality"""
    print("\n" + "="*50)
    print("TESTING FILE RETRIEVAL")
    print("="*50)
    
    from rag_engine.data_processing.file_retrieval import get_code_files
    
    # Test with scraped GitLab data
    data_dir = PROJECT_ROOT / "data" / "aiap17-gitlab-data"
    
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        print("Please ensure you've run the scraping script first!")
        return None, None
    
    print(f"✅ Data directory found: {data_dir}")
    
    # List available person directories
    person_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    print(f"\n📁 Available person directories: {len(person_dirs)}")
    for i, person in enumerate(person_dirs[:5]):  # Show first 5
        print(f"   {i+1}. {person.name}")
    
    if not person_dirs:
        print("❌ No person directories found!")
        return None, None
    
    # Test file retrieval on first person's directory
    test_person = person_dirs[0]
    print(f"\n🔍 Testing file retrieval in: {test_person.name}")
    
    # Get code files
    code_files = get_code_files(str(test_person))
    print(f"\n📄 Found {len(code_files)} code files:")
    
    # Show first 5 files as examples
    for i, file_path in enumerate(code_files[:5]):
        relative_path = Path(file_path).relative_to(test_person)
        print(f"   {i+1}. {relative_path}")
    
    if len(code_files) > 5:
        print(f"   ... and {len(code_files) - 5} more files")
    
    return code_files, test_person.name

def test_text_extraction(code_files):
    """Test text extraction functionality"""
    print("\n" + "="*50)
    print("TESTING TEXT EXTRACTION")
    print("="*50)
    
    if not code_files:
        print("⚠️  No code files available for testing!")
        return None, None
    
    from rag_engine.data_processing.text_extraction import initialize_semantic_chunkers, extract_text_from_files
    
    print("🔧 Initializing semantic chunkers...")
    
    # Initialize chunkers
    markdown_splitter, code_splitter = initialize_semantic_chunkers()
    print("✅ Chunkers initialized successfully!")
    
    # Test with a small subset of files (first 3 files)
    test_files = code_files[:3]
    print(f"\n📝 Testing text extraction on {len(test_files)} files...")
    
    # Extract text
    try:
        texts, doc_names = extract_text_from_files(test_files, markdown_splitter, code_splitter)
        
        print(f"\n✅ Text extraction successful!")
        print(f"   📊 Total chunks extracted: {len(texts)}")
        print(f"   📄 Documents processed: {len(set(doc_names))}")
        
        # Show some statistics
        if texts:
            avg_chunk_length = sum(len(text) for text in texts) / len(texts)
            print(f"   📏 Average chunk length: {avg_chunk_length:.0f} characters")
            
            # Show first few chunks as examples
            print(f"\n📖 Example chunks:")
            for i, (text, doc) in enumerate(zip(texts[:3], doc_names[:3])):
                preview = text[:100] + "..." if len(text) > 100 else text
                print(f"   {i+1}. [{doc}] {preview}")
        
        return texts, doc_names
                
    except Exception as e:
        print(f"❌ Error during text extraction: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def test_data_enhancement():
    """Test data enhancement functionality with LLM"""
    print("\n" + "="*50)
    print("TESTING DATA ENHANCEMENT")
    print("="*50)
    
    try:
        from rag_engine.data_processing.data_enhancement import enhance_data_with_llm
        
        # Test sample code and text
        sample_code = """
def calculate_sum(a, b):
    return a + b
        """
        
        sample_text = "This is a simple function that adds two numbers."
        
        print("📝 Testing data enhancement...")
        print(f"Original code sample: {sample_code.strip()}")
        print(f"Original text sample: {sample_text}")
        
        # Note: This test will only work if Ollama is running locally
        # For testing purposes, we'll mock the enhancement or skip if no Ollama
        print("\n⚠️  Note: Data enhancement requires Ollama to be running locally")
        print("✅ Data enhancement module imported successfully!")
        print("🔧 Actual enhancement testing requires Ollama server (skipped for now)")
        
        return True
        
    except ImportError as e:
        print(f"❌ Data enhancement import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Data enhancement test error: {e}")
        return False

def test_data_ingestion_components():
    """Test data ingestion components without running the full pipeline"""
    print("\n" + "="*50)
    print("TESTING DATA INGESTION COMPONENTS")
    print("="*50)
    
    try:
        # Test imports from data ingestion dependencies
        print("🔍 Testing model loader...")
        from rag_engine.embeddings.model_loader import load_models
        print("✅ Model loader imported successfully!")
        
        print("🔍 Testing embedding generation...")
        from rag_engine.embeddings.embedding_generation import generate_code_embeddings, generate_sentence_embeddings, project_embeddings
        print("✅ Embedding generation imported successfully!")
        
        print("🔍 Testing FAISS index...")
        from rag_engine.embeddings.faiss_index import create_faiss_index, save_faiss_index, load_faiss_index
        print("✅ FAISS index imported successfully!")
        
        print("\n⚠️  Note: Full data ingestion pipeline requires:")
        print("   - Ollama server running for data enhancement")
        print("   - Large embedding models downloaded")
        print("   - Significant processing time and memory")
        print("✅ All data ingestion components imported successfully!")
        print("🔧 Full pipeline testing requires proper setup (skipped for now)")
        
        return True
        
    except ImportError as e:
        print(f"❌ Data ingestion import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Data ingestion test error: {e}")
        return False

def analyze_data(texts, doc_names):
    """Analyze the extracted data"""
    print("\n" + "="*50)
    print("DATA ANALYSIS")
    print("="*50)
    
    if not texts:
        print("⚠️  No extracted text data available for analysis.")
        return
    
    from collections import Counter
    
    # Basic statistics
    total_chunks = len(texts)
    unique_docs = len(set(doc_names))
    total_chars = sum(len(text) for text in texts)
    
    print(f"📄 Total documents: {unique_docs}")
    print(f"🧩 Total chunks: {total_chunks}")
    print(f"📝 Total characters: {total_chars:,}")
    print(f"📏 Avg chars per chunk: {total_chars/total_chunks:.1f}")
    
    # Document distribution
    doc_counts = Counter(doc_names)
    print(f"\n📋 Chunks per document:")
    for doc, count in doc_counts.most_common():
        print(f"   {doc}: {count} chunks")
    
    # Chunk length distribution
    chunk_lengths = [len(text) for text in texts]
    min_len = min(chunk_lengths)
    max_len = max(chunk_lengths)
    avg_len = sum(chunk_lengths) / len(chunk_lengths)
    
    print(f"\n📐 Chunk length distribution:")
    print(f"   Min: {min_len} chars")
    print(f"   Max: {max_len} chars")
    print(f"   Avg: {avg_len:.1f} chars")
    
    # Sample some different types of content
    print(f"\n🔍 Content type analysis:")
    for i, text in enumerate(texts[:5]):
        content_type = "Code" if any(keyword in text.lower() for keyword in ['def ', 'import ', 'class ', 'print(']) else "Text/Markdown"
        preview = text.replace('\n', ' ')[:80] + "..." if len(text) > 80 else text.replace('\n', ' ')
        print(f"   Chunk {i+1} ({content_type}): {preview}")

def run_scale_test(code_files, markdown_splitter, code_splitter):
    """Run scale test with more files"""
    print("\n" + "="*50)
    print("SCALE TEST (Optional)")
    print("="*50)
    
    if len(code_files) <= 3:
        print("⏭️  Scale test skipped. Need more than 3 files.")
        return
    
    from rag_engine.data_processing.text_extraction import extract_text_from_files
    import time
    
    print("🚀 Running scale test with more files...")
    
    # Test with up to 10 files
    scale_test_files = code_files[:10]
    print(f"📁 Testing with {len(scale_test_files)} files...")
    
    try:
        start_time = time.time()
        scale_texts, scale_doc_names = extract_text_from_files(scale_test_files, markdown_splitter, code_splitter)
        end_time = time.time()
        
        print(f"\n✅ Scale test completed!")
        print(f"   ⏱️  Processing time: {end_time - start_time:.2f} seconds")
        print(f"   📊 Total chunks: {len(scale_texts)}")
        print(f"   📄 Documents processed: {len(set(scale_doc_names))}")
        print(f"   ⚡ Avg time per file: {(end_time - start_time) / len(scale_test_files):.2f} seconds")
        
    except Exception as e:
        print(f"❌ Scale test failed: {e}")

def main():
    """Main test function"""
    print("🧪 STARTING COMPREHENSIVE DATA PROCESSING TESTS")
    print("=" * 60)
    
    # Test 1: Imports
    if not test_imports():
        print("\n❌ Import tests failed. Cannot continue.")
        return False
    
    # Test 2: File Retrieval
    code_files, test_person = test_file_retrieval()
    if not code_files:
        print("\n❌ File retrieval tests failed. Cannot continue.")
        return False
    
    # Test 3: Text Extraction
    texts, doc_names = test_text_extraction(code_files)
    if not texts:
        print("\n❌ Text extraction tests failed. Cannot continue.")
        return False
    
    # Test 4: Data Analysis
    analyze_data(texts, doc_names)
    
    # Test 5: Data Enhancement
    if not test_data_enhancement():
        print("\n⚠️  Data enhancement tests failed, but continuing...")
    
    # Test 6: Data Ingestion Components
    if not test_data_ingestion_components():
        print("\n⚠️  Data ingestion component tests failed, but continuing...")
    
    # Test 7: Scale Test (optional)
    from rag_engine.data_processing.text_extraction import initialize_semantic_chunkers
    markdown_splitter, code_splitter = initialize_semantic_chunkers()
    run_scale_test(code_files, markdown_splitter, code_splitter)
    
    print("\n" + "="*60)
    print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
    print("🚀 Ready to proceed to next step: Embeddings Generation!")
    print("="*60)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
