# 📘 README: Custom RAG Engine – Capstone (AI Singapore AIAP Batch 17)

> Lightweight, modular Retrieval-Augmented Generation (RAG) system for internal document QA, enhanced with local LLMs, LangChain routing, and Kubernetes deployment.

---

## 🎯 Objective

Build a secure, containerized RAG pipeline that can:
- Parse, index, and retrieve answers from markdowns, PDFs, and structured documents
- Answer complex queries via vector similarity and pandas-based table reasoning
- Route fallback queries to self-reflective evaluators to improve response quality
- Deploy seamlessly on Kubernetes with Helm-ready configs for scalable infrastructure

---

## 📂 Project Structure

| Component                            | Description                                                                 |
|-------------------------------------|-----------------------------------------------------------------------------|
| `scrap_repos.ipynb`                 | Scrapes and parses GitLab repositories (markdowns/code)                    |
| `scrape_gitlab_files.ipynb`         | Extracts relevant files and tags with metadata                             |
| `mini-project.yml`                  | Environment spec with LangChain, HuggingFace, Ollama                       |
| `RAG bot.zip`                       | Full RAG source: Streamlit UI, Ollama backend, LangChain chains            |
| `kubernetes/deployment.yaml`        | Kubernetes deployment manifest for app container                           |
| `kubernetes/service.yaml`           | Service definition for Streamlit UI (ClusterIP or LoadBalancer)            |
| `kubernetes/ollama-deployment.yaml` | Deployment manifest for running Ollama LLM in Kubernetes                   |
| `Capstone_Presentation_Deck.pdf`    | System architecture, innovations, use cases, and reasoning pipeline        |

---

## 🧱 System Architecture

- **Frontend:** Streamlit chat + file upload UI
- **Backend:** Local LLaMA 3.1 Instruct via Ollama
- **Retriever Logic:** FAISS + HuggingFace Instructor Embeddings
- **Routing:** LangChain with:
  - Metadata filtering
  - Conversational memory
  - Pandas agent for tabular queries
  - Self-reflective fallback agents

---

## 🔍 Features

| Feature                    | Description                                                                       |
|---------------------------|-----------------------------------------------------------------------------------|
| 📎 File Ingestion         | Upload PDFs, markdowns, or CSVs                                                   |
| 🧩 Auto Chunking          | Uses LangChain recursive chunking with metadata (repo, path, file type, etc.)     |
| 📚 Mixed QA Modes         | Table querying via pandas agent, retrieval QA via vector similarity               |
| 🔁 Self-checker Agent     | Auto-checks LLM responses and invokes fallback logic when confidence is low       |
| 🚀 Kubernetes Ready       | Includes YAML for deployment, service exposure, and Ollama container integration  |

---

## ⚙️ Deployment Instructions

### 🔧 Local Development

```bash
# Prerequisites: Python 3.11+, ollama, faiss-cpu
conda create -n rag-bot python=3.11
pip install -r requirements.txt

# Start LLM
ollama run llama3:instruct

# Run Streamlit app
streamlit run app.py
```

### ☁ Kubernetes Deployment

```bash
# Deploy backend and Ollama LLM (ensure image is pushed to private/public registry)
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/service.yaml
kubectl apply -f kubernetes/ollama-deployment.yaml
```

> Tip: Add ingress or LoadBalancer as needed for production access.

---

## ✅ Evaluation

- Tested with internal HR-style queries (trend, breakdown, detail, list)
- Accurate retrieval across multi-format files
- Supports structured YAML outputs for downstream system integration
- Supports containerized, air-gapped deployment

---

## 📌 Conclusion

This capstone project successfully demonstrates:
- Secure, local document QA with self-hosted LLMs
- Robust retrieval augmented with pandas logic
- Deployment flexibility (local, docker, or K8s)
- Strong UX through Streamlit and modular chain logic

---

## 🛠 Skills & Tools Used

- Python · LangChain · Streamlit · FAISS · Ollama · HuggingFace
- Pydantic · YAML · Kubernetes · FastAPI · Docker
- Metadata filtering · Vector search · Self-evaluating agent chains

---

## 📚 References

1. https://github.com/langchain-ai/langchain
2. https://ollama.com
3. https://streamlit.io
4. https://python.langchain.com
5. https://github.com/hkunlp/instructor-embedding

---
