# 🧠 ML Guru - RAG Chatbot

**A production-ready RAG chatbot for Machine Learning & Deep Learning**  
Built with LangChain, LangGraph, FAISS, Groq LLaMA3, and Streamlit.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3+-green.svg)](https://python.langchain.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red.svg)](https://streamlit.io/)

## ✨ Features
- 🚀 **RAG Architecture** - Retrieval Augmented Generation from your personal documents
- 📚 **Multi-format Support** - Reads `.txt`, `.md`, and `.pdf` files
- 🧠 **Advanced Retrieval** - MMR (Maximum Marginal Relevance) for diverse results
- 💾 **Persistent FAISS Index** - Build once, load instantly on subsequent runs
- 🎨 **Modern UI** - Beautiful dark-themed Streamlit interface
- 🔄 **Session Memory** - LangGraph checkpoints for conversation history
- 📊 **Source Tracking** - See exactly which documents answers come from

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| LLM |  llama-3.3-70b-versatile (via LangChain) |
| Vector Store | FAISS (local) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Framework | LangChain + LangGraph |
| Frontend | Streamlit |
| Package Manager | `uv` (or pip) |

## 📦 Installation

### Prerequisites
- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- Groq API key ([get one here](https://console.groq.com/))

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/BivorAryal/RAG_CHATBOT.git
cd RAG_CHATBOT
```
2. **Create virtual environment**
```bash
# Using uv (fast)
uv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Using pip
python -m venv venv
# activate as above
```
3. **Install dependencies**
```bash
# Using uv (from pyproject.toml)
uv pip install -e .

# Using pip (from requirements.txt)
pip install -r requirements.txt
```
4. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```
5. **Configure document paths**
Edit `rag.backend.py` and update `ML_CONTENT_PATHS` to point to your ML/DL study materials:
```python
ML_CONTENT_PATHS = [
    r"C:\Your\Path\To\ML\Documents",
    r"D:\Another\Folder\With\PDFs",
]
```
## 🚀 Usage
### Quick Launch (Windows)
```bash
run_chatbot.bat
```
### Manual Launch (All platforms)
```bash
streamlit run rag.frontend.py
```
### First Run
- Builds FAISS index from your documents (1-3 minutes depending on document count)
- Downloads embedding model (once, ~80MB)
- Subsequent runs load in <5 seconds

## 📁 Project Structure
```text
RAG_CHATBOT/
├── rag.backend.py           # RAG pipeline (LangGraph + FAISS + Groq)
├── rag.frontend.py          # Streamlit UI
├── run_chatbot.bat      # Windows launcher
├── pyproject.toml       # Project metadata & dependencies
├── requirements.txt     # pip dependencies (optional)
├── .env                 # API keys (create from .env.example)
├── ml_rag_faiss_index/  # FAISS index storage (auto-created)
└── .gitignore          # Python standard ignores
```
## 🎯 How It Works
```text
User Question → LangGraph → Retrieve (FAISS) → Generate (Groq) → Answer + Sources
                    ↑                              ↓
              Conversation History ←── Context ────┘
```
1. Embedding - Your documents are chunked and embedded with all-MiniLM-L6-v2
2. Retrieval - FAISS finds top-k relevant chunks (MMR for diversity)
3. Generation - Groq's LLaMA 3.3 answers using ONLY retrieved context
4. Memory - LangGraph checkpointing maintains conversation state
---
## 🔧 Configuration
|Parameter|	Default|	Description|
|---------|------------|------------|
`chunk_size`|	800|	Document chunk size
`chunk_overlap`	150|	Overlap between chunks
`k` (retrieval)|	5	|Number of chunks retrieved
`temperature`|	0.3	|LLM creativity (0.0-1.0)
`model`	|llama-3.3-70b-versatile	|Groq model
---
Adjust in `rag.backend.py`.

## 🐛 Troubleshooting
**"No documents loaded" error**

→ Check ML_CONTENT_PATHS exist and contain .txt, .md, or .pdf files

**FAISS index rebuild needed**

→ Click "Rebuild FAISS Index" in sidebar or set force_rebuild=True

**Groq API key errors**

→ Verify GROQ_API_KEY in .env file

**Slow first run?**

→ Normal. FAISS indexing + model download takes time. Subsequent runs are fast.
---
## 📈 Performance Tips
- Place documents in fast SSD storage for faster indexing
- Use .txt files for faster loading (PDFs are slower)
- Keep chunk_size between 500-1000 for optimal results
- The FAISS index persists between runs - no need to rebuild

## 🤝 Contributing
Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📄 License
MIT License - feel free to use, modify, and distribute.

## 🙏 Acknowledgements
- LangChain for RAG framework
- Groq for fast LLM inference
- Streamlit for beautiful UI
- Sentence Transformers for embeddings
---
### Built with ❤️ by Bivor | GitHub

```text

## 🔍 Final Checklist Before GitHub Push

✅ **Security** - Ensure `.env` is in `.gitignore` (never commit API keys)  
✅ **Paths** - Your `ML_CONTENT_PATHS` use absolute paths - make this configurable via `.env`  
✅ **Cross-platform** - The batch file is Windows-only; consider adding a shell script for Linux/Mac  
✅ **Dependencies** - Decide on `requirements.txt` vs `pyproject.toml` (I'd keep both for flexibility)

## 🚀 Quick Verification Commands

Run these to ensure everything works:

# Check Python imports
python -c "import langchain, langgraph, faiss, streamlit, groq; print('✅ All imports OK')"

# Test backend independently
python -c "from rag.backend import MLRagChatbot; print('✅ Backend loads')"

# Launch manually
streamlit run rag. frontend.py
```


