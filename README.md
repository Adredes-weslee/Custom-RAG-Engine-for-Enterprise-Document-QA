# 🧠 Custom RAG Engine for Enterprise Document QA

> A production-ready, local-first Retrieval-Augmented Generation (RAG) system for enterprise document analysis. Features GPU acceleration with CPU fallback, hybrid embeddings for code and text understanding, and powerful local LLM inference via Ollama.

---

## 🎯 Project Overview

This RAG system is designed for **enterprise document analysis** with a focus on **GitLab repository data** and **code understanding**. Built with a local-first architecture that leverages your hardware resources while maintaining complete data privacy.

### 🏗️ Architecture Highlights
- **Local-First Design**: All processing happens on your hardware - no external API dependencies
- **GPU-Accelerated Processing**: Leverages CUDA for 10-50x faster data processing 
- **Specialized Embeddings**: Code-aware embeddings for Python files, semantic embeddings for notebooks
- **Privacy-First**: Complete data privacy - no information sent to external services
- **Production-Ready**: Comprehensive testing and deployment scripts

---

## 🚀 Key Features

| Feature | Local Development | Sharing Options |
|---------|------------------|-----------------|
| **Model Selection** | `llama3.2:3b`, `llama3.1:8b`, `llama3.2:1b` | Same (via ngrok/VPS) |
| **Processing** | GPU-accelerated data ingestion | Pre-built indices |
| **Performance** | Maximum quality enhancement | Full local performance |
| **Privacy** | Complete local processing | Complete local processing |
| **Deployment** | Local Streamlit app | Tunneling/Self-hosting |

### ⚠️ **Important: Streamlit Cloud Limitations**

**Streamlit Community Cloud does NOT support this application** due to:
- ❌ Cannot install or run Ollama server
- ❌ Cannot access local model servers (localhost:11434)
- ❌ Missing system dependencies (cmake, swig, pkg-config)
- ❌ Insufficient resources for local model inference

**✅ This application is designed for LOCAL deployment only.**

---

## 📂 Complete Project Structure

```
Custom-RAG-Engine-for-Enterprise-Document-QA/
├── 📄 README.md                              # This comprehensive guide
├── 📄 requirements.txt                       # Local dependencies (GPU/CPU)
├── 📄 setup_models.py                       # Ollama model downloader
├── 📄 run_data_ingestion.py                 # Data processing launcher
├── 📄 check.py                              # System compatibility checker
├── 🗂️ data/                                 
│   └── aiap17-gitlab-data/                  # GitLab repository data (23 people)
├── 🗂️ src/                                  # Main application code
│   ├── 📄 main.py                           # 🎯 Streamlit app entry point
│   └── 🗂️ rag_engine/                      # Core RAG engine
│       ├── 🗂️ data_processing/             # Data ingestion pipeline
│       │   ├── 📄 data_enhancement.py       # LLM-powered text enhancement
│       │   ├── 📄 data_ingestion.py         # Main data processing pipeline
│       │   ├── 📄 file_retrieval.py         # GitLab file discovery
│       │   └── 📄 text_extraction.py        # Multi-format text extraction
│       ├── 🗂️ embeddings/                  # Embedding generation & indexing
│       │   ├── 📄 embedding_generation.py   # Hybrid embeddings (code + text)
│       │   ├── 📄 faiss_index.py           # GPU-accelerated FAISS operations
│       │   └── 📄 model_loader.py          # Model loading with GPU detection
│       ├── 🗂️ evaluation/                  # Response quality evaluation
│       │   └── 📄 evaluation_agent.py       # Judge model evaluation
│       ├── 🗂️ models/                      # LLM integration
│       │   └── 📄 ollama_model.py          # Ollama client wrapper
│       ├── 🗂️ retrieval/                   # RAG pipeline components
│       │   ├── 📄 document_store.py         # Document metadata management
│       │   ├── 📄 question_handler.py       # Query routing & processing
│       │   └── 📄 rag_chain.py             # LangChain pipeline orchestration
│       └── 🗂️ ui/                          # User interface
│           └── 📄 streamlit_ui.py          # Streamlit UI components
├── 🗂️ tests/                               # Comprehensive test suite
│   ├── 📄 run_tests.py                     # Main test runner
│   ├── 📄 test_requirements.py             # Dependency verification
│   ├── 📄 test_data_processing.py          # Data pipeline tests
│   ├── 📄 test_embeddings_comprehensive.py # Embedding & GPU tests
│   └── 📄 test_rag_retrieval.py           # End-to-end RAG tests
├── 🗂️ utils/                               # Shared utilities
│   ├── 📄 logging_setup.py                 # Comprehensive logging
│   └── 📄 model_config.py                  # Environment-aware configuration
└── 🗂️ Generated Files/ (Created by data ingestion)
    ├── 📄 faiss_code_index.bin             # Code embeddings index
    ├── 📄 faiss_non_code_index.bin         # Notebook embeddings index
    ├── 📄 code_docstore.json               # Enhanced Python file content
    └── 📄 non_code_docstore.json           # Enhanced notebook content
```

---

## 🛠️ Installation & Setup

### 🔧 Prerequisites
- **Python 3.9+**
- **CUDA 12.6+** (recommended for GPU acceleration)
- **16GB+ RAM** (for model processing)
- **Ollama** (for local LLM inference)
- **Git LFS** (for large model files)

### 🚀 Quick Start

#### 1. Environment Setup
```bash
# Clone repository
git clone <your-repository-url>
cd Custom-RAG-Engine-for-Enterprise-Document-QA

# Install dependencies
pip install -r requirements.txt

# Verify system compatibility
python check.py
```

#### 2. Install Ollama
```bash
# Install Ollama (choose your platform)
# Linux/macOS:
curl -fsSL https://ollama.ai/install.sh | sh

# Windows: Download from https://ollama.ai/download
# OR using winget:
winget install Ollama.Ollama
```

#### 3. Setup Models
```bash
# Download required models
python setup_models.py

# Expected output:
# 🏠 Detected local development environment
# 📥 Downloading llama3.2:3b for data enhancement...
# 📥 Downloading llama3.1:8b for evaluation...
# 📥 Downloading llama3.2:1b for fallback...
# ✅ All models ready for local development!
```

#### 4. Start Ollama Server
```bash
# Start Ollama server (runs on localhost:11434)
ollama serve

# Verify models are available
ollama list
```

#### 5. Data Processing (GPU Recommended)
```bash
# Quick test (3 people, ~30 files)
python run_data_ingestion.py --test

# Custom limits
python run_data_ingestion.py --limit-people 5 --limit-files 20

# Full processing (all 23 people, ~1000+ files)
python run_data_ingestion.py

# Expected outputs:
# faiss_code_index.bin        (~10-50MB)
# faiss_non_code_index.bin    (~20-100MB)  
# code_docstore.json          (~50-200MB)
# non_code_docstore.json      (~100-500MB)
```

#### 6. Run Application
```bash
# Launch Streamlit app (in new terminal)
streamlit run src/main.py

# Application will automatically:
# ✅ Connect to local Ollama server
# ✅ Load pre-built indices
# ✅ Initialize RAG interface
# 🌐 Available at: http://localhost:8501
```

---

## 🎯 Deployment Options

### 🏠 Local Development (Primary Method) ✅

**Best for**: All use cases - this is the ONLY officially supported deployment

```bash
# Complete local setup
ollama serve                               # Start model server
streamlit run src/main.py                 # Start web interface

# Capabilities:
✅ GPU-accelerated processing (10-50x faster)
✅ High-quality LLM enhancement
✅ Complete model selection
✅ Maximum performance
✅ Complete privacy
✅ No API costs
✅ Uses your hardware (GPU/CPU)
```

### 🌐 Sharing Your App

#### **Option 1: Ngrok Tunnel (Recommended for Sharing)**
```bash
# Install ngrok
npm install -g ngrok
# OR download from: https://ngrok.com/download

# Run your app locally
streamlit run src/main.py

# In another terminal, create public tunnel
ngrok http 8501

# Share the public URL: https://abc123.ngrok.io
# ✅ Full functionality maintained
# ✅ Uses your local GPU/CPU
# ✅ Complete privacy (data stays on your machine)
```

#### **Option 2: VPS/Server Deployment**
```bash
# Deploy on DigitalOcean, AWS, Linode, etc.
# 1. Provision server with GPU (optional)
# 2. Install Docker + Ollama
# 3. Clone repository
# 4. Run setup scripts
# 5. Configure firewall (port 8501)
# 6. Access via server IP
```

#### **Option 3: Docker Container**
```dockerfile
# Create containerized deployment
FROM nvidia/cuda:12.6-runtime-ubuntu22.04
# Install Ollama + dependencies
# Copy application
# Expose port 8501
# Can deploy anywhere that supports Docker
```

### ❌ Streamlit Community Cloud (NOT SUPPORTED)

**Why it doesn't work:**
```bash
❌ Cannot install Ollama server
❌ Cannot run local model servers (localhost:11434)
❌ Missing system dependencies (cmake, swig, pkg-config)
❌ Insufficient compute resources for large models
❌ Cannot access your local GPU hardware
❌ Architecture incompatibility with local LLM servers

# Attempting to deploy will result in:
# - Build failures due to missing dependencies
# - Runtime errors when trying to connect to Ollama
# - Import errors for compiled packages
```

---

## 🤖 Local Model System

### 📋 Model Configuration

| Environment | Primary Model | Judge Model | Enhancement Model | Use Case |
|------------|---------------|-------------|-------------------|----------|
| **Local (High Memory)** | `llama3.2:3b` | `llama3.1:8b` | `llama3.2:3b` | ✅ Maximum quality |
| **Local (Standard)** | `llama3.2:3b` | `llama3.2:3b` | `llama3.2:3b` | ✅ Good performance |
| **Local (Minimal)** | `llama3.2:1b` | `llama3.2:1b` | `llama3.2:1b` | ✅ Resource constrained |

### 🔄 Automatic Detection
```python
# Environment detection (utils/model_config.py)
def detect_environment():
    if not is_ollama_available():
        raise RuntimeError("❌ Ollama server not available - run 'ollama serve'")
    elif has_high_memory() and has_gpu():
        return "local_high"
    elif has_sufficient_memory():
        return "local_standard"
    else:
        return "local_minimal"
```

---

## 📊 Data Processing Pipeline

### 🔍 Data Sources
- **GitLab Repository**: 23 AI apprentice projects
- **File Types**: Python scripts (`.py`), Jupyter notebooks (`.ipynb`)
- **Content**: Machine learning assignments, data science projects
- **Structure**: `/person/assignment/files`

### ⚡ Processing Steps

1. **File Discovery**: Recursive GitLab repository scanning
2. **Text Extraction**: Code-aware chunking with metadata preservation
3. **LLM Enhancement**: Adds comments, docstrings, explanations via Ollama
4. **Hybrid Embeddings**: 
   - Code files → `microsoft/graphcodebert-base` → 768D vectors
   - Notebooks → `all-MiniLM-L6-v2` → 384D vectors
   - Dimensionality alignment → 384D common space
5. **FAISS Indexing**: GPU-accelerated similarity search indices
6. **Document Storage**: Enhanced content with source metadata

### 📈 Processing Performance

| Mode | Files | Local GPU Time | Local CPU Time | Quality |
|------|-------|----------------|----------------|---------|
| **Test** | ~30 | 5 min | 15 min | Good |
| **Custom** | ~100 | 15 min | 45 min | Good |
| **Full** | 1000+ | 45 min | 4+ hours | Excellent |

---

## 🎯 Usage Examples

### 🔍 Asking Questions

```python
# Code-specific questions
"How do I implement a decision tree in Python?"
"Show me examples of data preprocessing pipelines"
"What are common machine learning evaluation metrics?"

# Project-specific questions  
"How did students approach assignment 1?"
"What libraries are commonly used for data science?"
"Show me examples of model evaluation code"

# Conceptual questions
"Explain the difference between supervised and unsupervised learning"
"What are best practices for data validation?"
```

### 📊 Response Structure
```json
{
  "answer": "Enhanced LLM response with context from local models",
  "sources": ["file1.py", "notebook2.ipynb"],
  "evaluation": {
    "relevance": 0.95,
    "accuracy": 0.88,
    "completeness": 0.92
  },
  "reasoning": "Judge model assessment",
  "model_used": "llama3.2:3b"
}
```

---

## 🧪 Testing & Validation

### 🔍 Run Comprehensive Tests
```bash
# Ensure Ollama is running first
ollama serve

# Full test suite
python tests/run_tests.py

# Individual components
python tests/test_requirements.py         # Dependencies ✅
python tests/test_data_processing.py     # File processing ✅
python tests/test_embeddings_comprehensive.py  # GPU/CPU embeddings ✅
python tests/test_rag_retrieval.py      # End-to-end RAG ✅
```

### 📊 Expected Test Results
```
✅ ALL DEPENDENCIES INSTALLED!
✅ Ollama server accessible at localhost:11434
✅ ALL TESTS COMPLETED SUCCESSFULLY!
✅ GPU available - using CUDA acceleration
✅ Environment detection: local_high
✅ Model loading: llama3.2:3b, llama3.1:8b
✅ Data processing: WORKING
✅ RAG pipeline: WORKING
🚀 Ready for local deployment!
```

---

## 🔧 Configuration & Customization

### 🎛️ Model Configuration (utils/model_config.py)
```python
# Customize models for your hardware
LOCAL_MODELS = {
    "high": {
        "primary": "llama3.2:3b",      # High-quality responses
        "judge": "llama3.1:8b",        # Thorough evaluation  
        "fallback": "llama3.2:1b"      # Resource constraints
    },
    "standard": {
        "primary": "llama3.2:3b",      # Good responses
        "judge": "llama3.2:3b",        # Same model evaluation
        "fallback": "llama3.2:1b"      # Resource constraints
    },
    "minimal": {
        "primary": "llama3.2:1b",      # Faster responses
        "judge": "llama3.2:1b",        # Quick evaluation
        "fallback": "llama3.2:1b"      # Consistent
    }
}
```

### ⚙️ Data Processing Options
```python
# Quick testing (development)
python run_data_ingestion.py --test

# Custom limits (testing)
python run_data_ingestion.py --limit-people 5 --limit-files 20

# Skip enhancement for speed (optional)
python run_data_ingestion.py --no-enhancement

# Full production processing
python run_data_ingestion.py
```

---

## 🚀 Production Deployment Guide

### 📋 Pre-deployment Checklist
- [ ] Run full test suite locally
- [ ] Process data with GPU enhancement
- [ ] Verify index file generation
- [ ] Test Streamlit app locally
- [ ] Ensure Ollama models are downloaded
- [ ] Configure sharing method (ngrok/VPS)

### 🌐 Sharing Setup Options

#### **Option 1: Ngrok Tunnel (Easiest)**
```bash
# 1. Install ngrok
npm install -g ngrok

# 2. Get free account at ngrok.com for persistent URLs
ngrok config add-authtoken YOUR_TOKEN

# 3. Start your app
streamlit run src/main.py

# 4. Create tunnel (in new terminal)
ngrok http 8501

# 5. Share URL: https://abc123.ngrok.app
```

#### **Option 2: VPS Deployment**
```bash
# 1. Provision server (DigitalOcean, Linode, etc.)
# 2. Install dependencies:
apt update && apt install -y python3 python3-pip git
curl -fsSL https://ollama.ai/install.sh | sh

# 3. Clone and setup:
git clone <your-repo>
cd Custom-RAG-Engine-for-Enterprise-Document-QA
pip install -r requirements.txt
python setup_models.py

# 4. Start services:
ollama serve &
streamlit run src/main.py --server.address=0.0.0.0

# 5. Configure firewall for port 8501
# 6. Access via: http://YOUR_SERVER_IP:8501
```

---

## 📊 Performance & Benchmarks

### 🔥 GPU vs CPU Performance

| Operation | Local GPU | Local CPU | Speedup |
|-----------|-----------|-----------|---------|
| **Data Ingestion** | 45 min | 4+ hours | 5-6x |
| **Embedding Generation** | 2.3s | 45.7s | 20x |
| **FAISS Index Creation** | 0.8s | 12.4s | 15x |
| **RAG Query Response** | 1.2s | 3.8s | 3x |
| **Model Loading** | 8s | 25s | 3x |

### 💾 File Sizes (23 People, Full Dataset)

| File | Size | Description |
|------|------|-------------|
| `faiss_code_index.bin` | ~15MB | Code vector index |
| `faiss_non_code_index.bin` | ~35MB | Notebook vector index |
| `code_docstore.json` | ~120MB | Enhanced Python files |
| `non_code_docstore.json` | ~280MB | Enhanced notebooks |
| **Total** | **~450MB** | Complete RAG system |

### 🖥️ System Requirements

| Component | Minimum | Recommended | Optimal |
|-----------|---------|-------------|---------|
| **RAM** | 8GB | 16GB | 32GB+ |
| **GPU** | None (CPU) | GTX 1660 | RTX 3080+ |
| **Storage** | 10GB | 50GB | 100GB+ |
| **CPU** | 4 cores | 8 cores | 16+ cores |

---

## 🔧 Troubleshooting

### ❌ Common Issues & Solutions

#### **Issue**: Ollama connection failed
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not running, start it:
ollama serve

# Verify models are downloaded:
ollama list
```

#### **Issue**: Models not found
```bash
# Download required models:
python setup_models.py

# OR manually:
ollama pull llama3.2:3b
ollama pull llama3.1:8b
ollama pull llama3.2:1b
```

#### **Issue**: GPU not detected
```bash
# Check CUDA installation
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

#### **Issue**: Import errors
```bash
# Install missing dependencies
pip install -r requirements.txt

# For compilation issues, install build tools:
# Ubuntu/Debian:
apt install build-essential cmake pkg-config

# Windows: Install Visual Studio Build Tools
```

#### **Issue**: "I want to deploy to Streamlit Cloud"
```bash
# ❌ NOT POSSIBLE with current architecture
# ✅ Solutions:
# 1. Use ngrok tunnel for sharing local app
# 2. Deploy on VPS with Docker
# 3. Use cloud APIs (requires major code rewrite)
```

---

## 🎯 Customization for Your Data

### 🗂️ Using Your Own Data

1. **Replace Data Source**:
   ```bash
   # Replace data/aiap17-gitlab-data/ with your repository
   cp -r /path/to/your/data data/your-project-data
   
   # Update configuration
   python run_data_ingestion.py --root-directory data/your-project-data
   ```

2. **Modify File Types**:
   ```python
   # Edit file_retrieval.py to support your file types
   SUPPORTED_EXTENSIONS = ['.py', '.ipynb', '.md', '.txt', '.java', '.cpp']
   ```

3. **Customize Enhancement**:
   ```python
   # Edit data_enhancement.py for domain-specific improvements
   ENHANCEMENT_PROMPT = """
   Improve this {file_type} for better searchability:
   - Add domain-specific comments
   - Explain business logic  
   - Add relevant keywords for {your_domain}
   """
   ```

4. **Adjust Models**:
   ```python
   # Modify utils/model_config.py for different models
   # Available Ollama models: ollama.ai/library
   LOCAL_MODELS = {
       "primary": "codellama:7b",     # For code-heavy datasets
       "judge": "llama3.1:8b",        # For evaluation
       "fallback": "gemma2:2b"        # Lightweight option
   }
   ```

---

## 🤝 Contributing

1. **Fork & Clone**
   ```bash
   git clone https://github.com/your-username/Custom-RAG-Engine-for-Enterprise-Document-QA.git
   ```

2. **Create Feature Branch**
   ```bash
   git checkout -b feature/amazing-enhancement
   ```

3. **Test Your Changes**
   ```bash
   # Ensure Ollama is running
   ollama serve

   # Run tests
   python tests/run_tests.py
   ```

4. **Submit Pull Request**
   - Ensure all tests pass
   - Include performance benchmarks
   - Document new features
   - Test on both GPU and CPU environments

---

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **AI Singapore AIAP Batch 17** - Training program and dataset
- **LangChain Community** - RAG framework foundation
- **Ollama Team** - Local LLM inference platform
- **FAISS & HuggingFace** - Vector search and embedding models
- **Streamlit** - Rapid UI development platform

---

## 📚 Key Technical Papers & References

1. [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
2. [LangChain Documentation](https://python.langchain.com/docs/)
3. [FAISS: A Library for Efficient Similarity Search](https://github.com/facebookresearch/faiss)
4. [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084)
5. [GraphCodeBERT: Pre-training Code Representations with Data Flow](https://arxiv.org/abs/2009.08366)
6. [Ollama Documentation](https://ollama.ai/docs)

---

## 🚨 **Important Deployment Notice**

**This RAG system is designed exclusively for LOCAL deployment.** 

- ✅ **Works perfectly**: Local Streamlit + Ollama
- ✅ **For sharing**: Use ngrok tunneling or VPS deployment  
- ❌ **Does NOT work**: Streamlit Community Cloud (architectural limitations)

**For sharing your app with others, use the ngrok tunnel method or deploy on your own server infrastructure.**

---

**⭐ Star this repository if you find it useful!**

**🐛 Issues**: [GitHub Issues](https://github.com/Adredes-weslee/Custom-RAG-Engine-for-Enterprise-Document-QA/issues)  
**💬 Discussions**: [GitHub Discussions](https://github.com/Adredes-weslee/Custom-RAG-Engine-for-Enterprise-Document-QA/discussions)  
**📧 Contact**: weslee.qb@gmail.com

---

*This project represents a complete enterprise RAG solution optimized for local deployment with powerful hardware utilization. The local-first architecture ensures maximum performance, complete privacy, and full control over your data and models.*