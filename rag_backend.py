"""
============================================================
  ML & Deep Learning RAG Chatbot Backend
  Built with: LangChain | LangGraph | FAISS | Groq
============================================================
  Author: Bivor
  Stack : LangChain + LangGraph + FAISS + Groq LLaMA3
============================================================
"""

# ── 0. Imports ────────────────────────────────────────────
import os
from pathlib import Path
from typing import Annotated, TypedDict, List
from dotenv import load_dotenv
# LangChain – Document loaders
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
)
# LangChain – Splitters & Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# LangChain – Core
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

# LangChain – LLM (Groq)
from langchain_groq import ChatGroq

# LangGraph
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

# ── 1. Constants & Paths ──────────────────────────────────
FAISS_INDEX_PATH = "ml_rag_faiss_index"
GROQ_API_KEY     = os.getenv("GROQ_API_KEY")

# Cache the embedding model in a module-level variable
# so it is loaded ONCE and reused across all calls
_EMBEDDINGS = None

def get_embeddings() -> HuggingFaceEmbeddings:
    """Return a cached HuggingFaceEmbeddings instance."""
    global _EMBEDDINGS
    if _EMBEDDINGS is None:
        print("🔄  Loading embedding model (first time only)…")
        _EMBEDDINGS = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        print("✅  Embedding model loaded.")
    return _EMBEDDINGS

# Paths on your Desktop that contain ML/DL content
ML_CONTENT_PATHS = [
    r"C:\Users\bivor\OneDrive\Desktop\Gen AI Practice\10.RAG",
    r"C:\Users\bivor\OneDrive\Desktop\Gen AI Practice\9.RAG_BASED_APPLICATION",
    r"C:\Users\bivor\OneDrive\Desktop\ML Projects Oct24\Types of ML with Examples",
    r"C:\Users\bivor\OneDrive\Desktop\Agentic_AI_Using_LangGraph",
    r"C:\Users\bivor\OneDrive\Desktop\Gen AI Practice\books",
]

# ── 2. LLM ────────────────────────────────────────────────
def get_llm() -> ChatGroq:
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        max_tokens=1024,
    )

# ── 3. Document Loading ───────────────────────────────────
def load_documents(paths: List[str]) -> list:
    """Load .txt, .md, and .pdf files from provided paths."""
    all_docs = []
    for raw_path in paths:
        p = Path(raw_path)
        if not p.exists():
            print(f"⚠️  Skipping (not found): {p}")
            continue
        print(f"📂  Loading: {p.name}")

        for glob_pattern, loader_cls in [
            ("**/*.txt", TextLoader),
            ("**/*.md",  TextLoader),
        ]:
            try:
                loader = DirectoryLoader(
                    str(p),
                    glob=glob_pattern,
                    loader_cls=loader_cls,
                    loader_kwargs={"encoding": "utf-8", "autodetect_encoding": True},
                    silent_errors=True,
                    show_progress=False,
                    exclude=["**/.venv/**", "**/venv/**", "**/__pycache__/**",
                             "**/.git/**", "**/*.py", "**/*.ipynb"],
                )
                docs = loader.load()
                if docs:
                    print(f"   ✅ {glob_pattern}: {len(docs)} docs")
                all_docs.extend(docs)
            except Exception as e:
                print(f"   ⚠️  {glob_pattern} error: {e}")

        for pdf_file in p.rglob("*.pdf"):
            # Skip PDFs inside .venv or node_modules
            if any(skip in str(pdf_file) for skip in [".venv", "venv", "__pycache__"]):
                continue
            try:
                loader = PyPDFLoader(str(pdf_file))
                docs = loader.load()
                print(f"   ✅ PDF {pdf_file.name}: {len(docs)} pages")
                all_docs.extend(docs)
            except Exception as e:
                print(f"   ⚠️  PDF {pdf_file.name}: {e}")

    print(f"\n📄  Total docs loaded: {len(all_docs)}")
    return all_docs

# ── 4. FAISS Index ────────────────────────────────────────
def build_vector_store(force_rebuild: bool = False) -> FAISS:
    """Build FAISS index once and persist to disk. Reload on subsequent runs."""
    index_path = Path(FAISS_INDEX_PATH)
    embeddings = get_embeddings()

    # ── Fast path: load existing index from disk ──────────
    if index_path.exists() and not force_rebuild:
        print("📦  Loading FAISS index from disk (fast)…")
        vs = FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        print("✅  FAISS index loaded.")
        return vs

    # ── Slow path: build from scratch (first run only) ───
    print("🔨  Building FAISS index (first run — will be fast next time)…")
    docs = load_documents(ML_CONTENT_PATHS)

    if not docs:
        raise ValueError(
            "No documents loaded. Check that ML_CONTENT_PATHS exist in rag_backend.py."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"✂️   {len(chunks)} chunks created")

    vs = FAISS.from_documents(chunks, embeddings)
    vs.save_local(FAISS_INDEX_PATH)
    print(f"💾  FAISS index saved → '{FAISS_INDEX_PATH}/' (instant load next time)")
    return vs

# ── 5. Retriever ──────────────────────────────────────────
def get_retriever(vector_store: FAISS, k: int = 5):
    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": 20, "lambda_mult": 0.7},
    )

# ── 6. Prompt ─────────────────────────────────────────────
RAG_SYSTEM_PROMPT = """You are an expert ML & Deep Learning tutor named **ML Guru**.
Answer questions using ONLY the provided context from the user's personal study materials.

Guidelines:
- Be clear, structured, and educational.
- Use bullet points and examples when helpful.
- If the context is insufficient, say: "I don't have enough info in your study materials on this."
- Never hallucinate. Stick strictly to the provided context.

Context from your study materials:
{context}
"""

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", RAG_SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])


# ── 7. LangGraph State ────────────────────────────────────
class RAGState(TypedDict):
    messages:  Annotated[list[BaseMessage], add_messages]
    question:  str
    context:   str
    answer:    str
    sources:   List[str]

# ── 8. LangGraph Graph ────────────────────────────────────
def build_rag_graph(retriever):
    llm = get_llm()

    def retrieve_node(state: RAGState) -> RAGState:
        docs = retriever.invoke(state["question"])
        parts, sources = [], []
        for i, doc in enumerate(docs, 1):
            src = doc.metadata.get("source", "Unknown")
            src_short = Path(src).name if src != "Unknown" else "Unknown"
            parts.append(f"[{i}] ({src_short})\n{doc.page_content}")
            if src_short not in sources:
                sources.append(src_short)
        return {**state, "context": "\n\n---\n\n".join(parts), "sources": sources}

    def generate_node(state: RAGState) -> RAGState:
        chain  = rag_prompt | llm | StrOutputParser()
        answer = chain.invoke({
            "question":     state["question"],
            "context":      state["context"],
            "chat_history": state["messages"][:-1],
        })
        return {**state, "answer": answer, "messages": [AIMessage(content=answer)]}

    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_edge(START,      "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile(checkpointer=InMemorySaver())


# ── 9. Public Chatbot Class ───────────────────────────────
class MLRagChatbot:
    def __init__(self, force_rebuild: bool = False):
        print("\n🚀  Initializing ML RAG Chatbot…")
        self.vector_store    = build_vector_store(force_rebuild=force_rebuild)
        self.retriever       = get_retriever(self.vector_store)
        self.graph           = build_rag_graph(self.retriever)
        self._thread_counter = 0
        print("✅  Chatbot ready!\n")

    def new_thread_id(self) -> str:
        self._thread_counter += 1
        return f"session-{self._thread_counter}"

    def chat(self, question: str, thread_id: str) -> dict:
        config = {"configurable": {"thread_id": thread_id}}
        result = self.graph.invoke(
            {
                "messages": [HumanMessage(content=question)],
                "question": question,
                "context":  "",
                "answer":   "",
                "sources":  [],
            },
            config=config,
        )
        return {"answer": result["answer"], "sources": result["sources"]}
