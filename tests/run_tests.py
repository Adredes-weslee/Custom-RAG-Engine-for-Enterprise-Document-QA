"""
Test Runner for RAG Engine

This script runs all tests in the correct order.

Usage: python run_tests.py
"""

import sys
import subprocess
from pathlib import Path

def run_test(test_name, test_file):
    """Run a single test and return success status"""
    print(f"\n🧪 Running {test_name}...")
    print("-" * 50)
    
    try:
        result = subprocess.run([sys.executable, test_file], 
                              capture_output=False, 
                              check=True)
        print(f"✅ {test_name} passed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {test_name} failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"❌ Error running {test_name}: {e}")
        return False

def main():
    """Run all tests"""
    test_dir = Path(__file__).parent
    tests = [
        ("Requirements Test", test_dir / "test_requirements.py"),
        ("Data Processing Test", test_dir / "test_data_processing.py"),
        ("Comprehensive Embeddings Test", test_dir / "test_embeddings_comprehensive.py"),
        ("RAG Retrieval Pipeline Test", test_dir / "test_rag_retrieval.py"),
    ]
      # Optional tests that can be run separately
    optional_tests = [
        # Chunking comparison functionality now integrated into comprehensive embeddings test
    ]
    
    print("🚀 STARTING ALL RAG ENGINE TESTS")
    print("=" * 60)
    
    passed = 0
    total = len(tests)
    
    # Run core tests
    for test_name, test_file in tests:
        if not test_file.exists():
            print(f"❌ Test file not found: {test_file}")
            continue
            
        if run_test(test_name, test_file):
            passed += 1
    
    # Run optional tests (failures don't affect overall result)
    print(f"\n📋 OPTIONAL TESTS:")
    print("-" * 30)
    
    for test_name, test_file in optional_tests:
        if test_file.exists():
            try:
                run_test(test_name, test_file)
                print(f"✅ {test_name} (optional) completed")
            except:
                print(f"⚠️  {test_name} (optional) failed - continuing...")
    
    print("\n" + "=" * 60)
    print(f"📊 CORE TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL CORE TESTS PASSED!")
        print("🚀 Ready to proceed with development!")
    else:
        print("⚠️  Some core tests failed. Please fix issues before proceeding.")
    
    print("=" * 60)
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)