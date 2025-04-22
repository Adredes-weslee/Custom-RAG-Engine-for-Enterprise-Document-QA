# 📘 README: Custom RAG Engine – Capstone (AI Singapore AIAP Batch 17)

> Lightweight, modular Retrieval-Augmented Generation (RAG) system for internal document QA, enhanced with self-hosted LLMs, LangChain routing, and Kubernetes-ready deployment.

---

## 🎯 Objective

Develop a production-grade RAG system that can:
- Parse and semantically index markdowns, PDFs, and tabular files
- Handle complex internal queries via LangChain chains and agents
- Retrieve precise, structured answers with metadata and self-reflection
- Deploy securely on Kubernetes with modular microservice architecture

---

## 📂 Project Structure

| Component                          | Description                                                       |
|-----------------------------------|-------------------------------------------------------------------|
| `src/`                             | All backend logic: embedding, RAG chain, model loading, agents    |
| `streamlit_ui.py`                 | Main UI app entrypoint for document upload + query interaction   |
| `start.sh`                        | Shell script for local development and streamlit run             |
| `mini-project.yml`                | Environment and evaluation routing spec (LangChain-based)        |
| `deployment.yaml` & `service.yaml`| Kubernetes manifests for deploying the app and exposing services |
| `dockerfiles/Dockerfile`         | Dockerfile for containerizing the RAG system                     |
| `scrape_gitlab_files.ipynb`       | Utility script for parsing GitLab files (markdown/code)          |
| `requirements.txt`                | Dependency list for pip install                                   |

---

## 🧱 System Architecture

- **Frontend:** Streamlit interface with drag-and-drop file upload and chat-style interaction
- **LLM Backend:** Local Ollama instance running `llama3:instruct`
- **Vector Store:** FAISS with HuggingFace `instructor-large` embeddings
- **RAG Chain:** LangChain-based orchestration with:
  - Metadata filtering
  - Pandas agent for tabular queries
  - Self-evaluation agent for fallback routing

---

## 🔍 Key Features

| Feature                    | Description                                                                 |
|---------------------------|-----------------------------------------------------------------------------|
| 📎 Upload & Ingest        | Ingests markdown, code, CSV, PDF documents                                  |
| 🧠 Smart QA Pipeline       | Automatically routes queries to retriever or tabular agent                  |
| 🧩 Chunking + Metadata     | Recursive chunking with custom tags (`repo`, `path`, `type`)                |
| 🔁 Feedback Agent          | Evaluates response quality and triggers fallback if needed                  |
| 🐳 Dockerized              | Dockerfile included for local builds and testing                            |
| ☁ Kubernetes Ready         | Manifests provided to run the full app and Ollama on K8s                    |

---

## ⚙️ Local Setup

```bash
# Install environment
conda create -n rag-bot python=3.11
pip install -r requirements.txt

# Start Ollama
ollama run llama3:instruct

# Launch app
bash start.sh
```

---

## ☁ Kubernetes Deployment

```bash
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
```

> Ollama deployment assumes separate node/container — recommended for GPU-backed environments.

---

## ✅ Evaluation Highlights

- Supports structured outputs via YAML parsing
- Evaluated on internal HR-style analytical queries
- Modular logic enables agent chaining and contextual reranking
- Suitable for offline deployment, enterprise firewalls, or secure air-gapped usage

---

## 📌 Conclusion

This capstone delivers a reusable, containerized Retrieval-Augmented Generation (RAG) system designed for internal enterprise knowledge retrieval. Emphasis is placed on:
- Retrieval precision via embeddings + metadata
- Reasoning across structured data (e.g. CSVs) with pandas agents
- UX and modular logic routing
- Scalable deployment with Kubernetes support

---

## 🛠 Tools & Skills

- Python · LangChain · FAISS · Ollama · Streamlit · Kubernetes · Docker  
- HuggingFace Embeddings · Self-Reflective Agents · YAML Routing

---

## 📚 References

1. https://github.com/langchain-ai/langchain  
2. https://ollama.com  
3. https://streamlit.io  
4. https://github.com/hkunlp/instructor-embedding  
5. https://python.langchain.com/docs/use_cases/question_answering/

---

