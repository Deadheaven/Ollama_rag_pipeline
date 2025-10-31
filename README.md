# 🧩 Local RAG Pipeline with Ollama + FAISS

A lightweight **Retrieval-Augmented Generation (RAG)** pipeline that runs **entirely locally** — no cloud APIs required.  
Built using **LangChain**, **FAISS**, and **Ollama** with open-source models like Qwen and Nomic.

---

## ✨ Features

- 📄 Load and index local PDF documents
- 🧠 Embed using Ollama’s `nomic-embed-text`
- 🔎 Retrieve context chunks via FAISS (no SQLite)
- 🤖 Query locally with Qwen, Mistral, or LLaMA models
- 💾 100% local — no API keys, no internet dependency

---

## 🧱 Tech Stack

| Component | Purpose |
|------------|----------|
| [LangChain](https://python.langchain.com/) | RAG and retrieval orchestration |
| [FAISS](https://github.com/facebookresearch/faiss) | Vector similarity search |
| [Ollama](https://ollama.ai) | Local LLM & embedding inference |
| [PyPDF](https://pypi.org/project/pypdf/) | PDF text extraction |

---

## ⚙️ Installation

```bash
# 1️⃣ Clone the repository
git clone https://github.com/<your-username>/rag-ollama-faiss.git
cd rag-ollama-faiss

# 2️⃣ Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3️⃣ Install dependencies
pip install -r requirements.txt

# 4️⃣ Start Ollama service (in another terminal)
ollama serve

# 5️⃣ Run the script
python rag_local_faiss.py
