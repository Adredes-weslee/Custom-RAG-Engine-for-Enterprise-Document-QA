# 🧠 Custom RAG Engine – Capstone (AI Singapore AIAP B17)

> A secure, containerized Retrieval-Augmented Generation (RAG) system for querying enterprise documents (code, markdown, and structured data), powered by open-source LLMs and hybrid embeddings.

---

## 🎯 Objective

Build a local, production-grade RAG system that:
- Parses internal documentation (markdown, CSV, PDFs, code)
- Retrieves semantically relevant chunks using hybrid embeddings
- Answers queries via local LLMs with table reasoning and fallback agents
- Deploys on Kubernetes using containerized backend/frontend services

---

## 📂 Project Structure

| File/Dir                         | Description                                                                |
|----------------------------------|----------------------------------------------------------------------------|
| `src/`                           | Main source code for RAG engine                                            |
| ├─ `streamlit_ui.py`            | Frontend: file uploader + chat interface                                  |
| ├─ `main.py`                    | Backend controller for loading models and chains                          |
| ├─ `rag_chain.py`               | LangChain pipeline orchestration (retrieval, routing, evaluation)         |
| ├─ `model_loader.py`            | Loads MiniLM + GraphCodeBERT for hybrid embedding                         |
| ├─ `embedding_generation.py`    | Generates embeddings using HuggingFace models                             |
| ├─ `data_ingestion.py`          | Handles document chunking + metadata tagging                              |
| ├─ `faiss_index.py`             | FAISS vector store setup + retrieval                                      |
| ├─ `evaluation_agent.py`        | LangChain-based self-reflective fallback agent                            |
| ├─ `question_handler.py`        | Pandas agent + YAML output formatting                                     |
| ├─ `document_store.py`          | Internal store for parsed document metadata                               |
| `requirements.txt`              | Project dependencies                                                      |
| `deployment.yaml`, `service.yaml`| Kubernetes manifests for deploying backend + Streamlit app                |
| `dockerfiles/Dockerfile`        | Docker container for RAG engine                                           |
| `mini-project.yml`              | Agent orchestration config                                                |
| `start.sh`                      | Convenience script to launch the app locally                              |

---

## 🧱 System Architecture

- **Frontend:** Streamlit UI for uploading files and submitting queries
- **Backend:** Ollama-hosted LLaMA 3.1 Instruct model
- **Embedding Strategy:** Hybrid embeddings using:
  - `all-MiniLM-L6-v2` for text/markdown
  - `microsoft/graphcodebert-base` for code files
- **Retriever:** FAISS vector store with LangChain retriever
- **Routing:** LangChain chains with:
  - Metadata filtering
  - Conversational memory
  - Pandas agent for structured table reasoning
  - Self-reflective evaluator fallback agent

---

## 🔍 Key Features

| Feature                      | Description                                                                 |
|-----------------------------|-----------------------------------------------------------------------------|
| 🗂️ Hybrid Embedding Loader  | Loads separate models for text and code (MiniLM + GraphCodeBERT)            |
| 🧩 Recursive Chunking        | Intelligent chunk splitting with file-level metadata                        |
| 🧠 Self-Evaluation Agent     | Re-evaluates weak answers and auto-routes to fallback chain                 |
| 📊 Tabular Reasoning Agent   | Uses pandas to answer table-style queries                                  |
| 🔒 Secure Deployment         | Local LLM inference via Ollama + containerized with Kubernetes manifests    |
| 🎛️ Multi-format Support     | Handles markdowns, CSVs, PDFs (via text extraction), and source code        |

---

## ⚙️ Deployment

### 🚀 Local Setup

```bash
# Install Ollama, FAISS, and Python 3.11+
ollama run llama3:instruct

conda create -n rag-bot python=3.11
pip install -r requirements.txt

# Run Streamlit app
streamlit run src/streamlit_ui.py
```

### ☁ Kubernetes Setup

```bash
# Apply manifests (Streamlit + Ollama)
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
```

---

## ✅ Evaluation

- Query types supported: breakdown, trend, list, correlation, and detail
- Auto-routes to tabular agent or fallback evaluator when applicable
- Produces YAML outputs for downstream integration
- Built for air-gapped or secure enterprise environments

---

## 📌 Conclusion

This capstone demonstrates a secure, extensible RAG system that can reason over diverse enterprise documentation (markdowns, tables, code) using hybrid embeddings and open-source LLMs. Modular LangChain routing and a polished Streamlit UI make it a powerful internal knowledge retrieval tool.

---

## 🛠 Skills & Tools Used

- Python · LangChain · Streamlit · FAISS · Ollama
- MiniLM + GraphCodeBERT (HuggingFace)
- Pydantic · YAML · Pandas Agents
- Kubernetes · Docker · Markdown & Code Indexing

---

## 📚 References

1. https://github.com/langchain-ai/langchain
2. https://ollama.com
3. https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
4. https://huggingface.co/microsoft/graphcodebert-base
5. https://streamlit.io

---
