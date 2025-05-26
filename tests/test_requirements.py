"""
Test Dependencies and Requirements

This script tests if all required dependencies are installed and working.
Run this before running the main data processing tests.

Usage: python test_requirements.py
"""

import sys
import importlib
from pathlib import Path

def test_dependency(package_name, import_name=None):
    """Test if a dependency is installed and importable"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'Unknown')
        print(f"✅ {package_name}: {version}")
        return True
    except ImportError as e:
        print(f"❌ {package_name}: Not installed or import error - {e}")
        return False

def main():
    """Test all required dependencies"""
    print("🔍 TESTING DEPENDENCIES")
    print("=" * 40)
    
    dependencies = [
        ('pandas', 'pandas'),
        ('requests', 'requests'),
        ('python-gitlab', 'gitlab'),
        ('langchain', 'langchain'),
        ('sentence-transformers', 'sentence_transformers'),
        ('faiss-cpu', 'faiss'),
        ('streamlit', 'streamlit'),
        ('python-dotenv', 'dotenv'),
        ('huggingface-hub', 'huggingface_hub'),
        ('transformers', 'transformers'),
        ('torch', 'torch'),
        ('numpy', 'numpy'),
    ]
    
    missing_deps = []
    
    for package_name, import_name in dependencies:
        if not test_dependency(package_name, import_name):
            missing_deps.append(package_name)
    
    print("\n" + "=" * 40)
    
    if missing_deps:
        print(f"❌ MISSING DEPENDENCIES: {len(missing_deps)}")
        print("\nTo install missing dependencies, run:")
        print("pip install " + " ".join(missing_deps))
        return False
    else:
        print("✅ ALL DEPENDENCIES INSTALLED!")
        print("🚀 Ready to run data processing tests!")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
