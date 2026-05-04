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
import logging
from pathlib import Path
from typing import Annotated, TypedDict, List
from datetime import datetime
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

# ── Logging Setup ─────────────────────────────────────────
def setup_logger(name: str, log_file: str = "rag_chatbot.log"):
    """Setup logger with both file and console handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

# Initialize logger
logger = setup_logger("RAG_Chatbot")

# ── 1. Constants & Paths ──────────────────────────────────
FAISS_INDEX_PATH = "ml_rag_faiss_index"
GROQ_API_KEY     = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    logger.error("GROQ_API_KEY not found in environment variables!")
    raise ValueError("GROQ_API_KEY is required. Please check your .env file.")

logger.info("=" * 60)
logger.info("ML RAG Chatbot Backend Initializing")
logger.info(f"FAISS Index Path: {FAISS_INDEX_PATH}")
logger.info(f"Python Version: {os.sys.version}")
logger.info("=" * 60)

# Cache the embedding model in a module-level variable
# so it is loaded ONCE and reused across all calls
_EMBEDDINGS = None

def get_embeddings() -> HuggingFaceEmbeddings:
    """Return a cached HuggingFaceEmbeddings instance."""
    global _EMBEDDINGS
    if _EMBEDDINGS is None:
        logger.info("Loading embedding model (first time only)...")
        try:
            _EMBEDDINGS = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
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
logger.info(f"Configured document paths: {len(ML_CONTENT_PATHS)} directories")

# ── 3. Document Loading ───────────────────────────────────
def load_documents(paths: List[str]) -> list:
    """Load .txt, .md, and .pdf files from provided paths."""
    logger.info("Starting document loading process")
    all_docs = []
    start_time = datetime.now()

    for raw_path in paths:
        p = Path(raw_path)
        if not p.exists():
            logger.warning(f"Path not found, skipping: {p}")
            continue
        logger.info(f"Loading documents from: {p.name}")

        # Load text and markdown files
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
                    logger.info(f"  Loaded {len(docs)} {glob_pattern} files from {p.name}")
                all_docs.extend(docs)
            except Exception as e:
                logger.error(f"  Error loading {glob_pattern} from {p.name}: {e}")

        # Load PDF files
        pdf_count = 0

        for pdf_file in p.rglob("*.pdf"):
            # Skip PDFs inside .venv or node_modules
            if any(skip in str(pdf_file) for skip in [".venv", "venv", "__pycache__"]):
                continue
            try:
                loader = PyPDFLoader(str(pdf_file))
                docs = loader.load()
                logger.debug(f"  Loaded PDF: {pdf_file.name} ({len(docs)} pages)")
                all_docs.extend(docs)
            except Exception as e:
                logger.error(f"  Error loading PDF {pdf_file.name}: {e}")

        if pdf_count > 0:
                    logger.info(f"  Loaded {pdf_count} PDF files from {p.name}")

    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(f"Document loading complete: {len(all_docs)} documents loaded in {elapsed:.2f}s")
    return all_docs

# ── 4. FAISS Index ────────────────────────────────────────
def build_vector_store(force_rebuild: bool = False) -> FAISS:
    """Build FAISS index once and persist to disk. Reload on subsequent runs."""
    index_path = Path(FAISS_INDEX_PATH)
    embeddings = get_embeddings()

    # ── Fast path: load existing index from disk ──────────
    if index_path.exists() and not force_rebuild:
        logger.info("Loading existing FAISS index from disk...")
        try:
            vs = FAISS.load_local(
                FAISS_INDEX_PATH,
                embeddings,
                allow_dangerous_deserialization=True,
            )
            logger.info("FAISS index loaded successfully (fast path)")
            return vs
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            logger.info("Will rebuild index instead...")


    # ── Slow path: build from scratch (first run only) ───
    logger.info("Building FAISS index from scratch...")
    start_time = datetime.now()

    docs = load_documents(ML_CONTENT_PATHS)

    if not docs:
        logger.error("No documents loaded. Check ML_CONTENT_PATHS configuration.")
        raise ValueError(
            "No documents loaded. Check that ML_CONTENT_PATHS exist in rag_backend.py."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    logger.info(f"Created {len(chunks)} chunks from {len(docs)} documents")

    logger.info("Generating embeddings and building FAISS index...")
    vs = FAISS.from_documents(chunks, embeddings)

    logger.info(f"Saving FAISS index to {FAISS_INDEX_PATH}...")
    vs.save_local(FAISS_INDEX_PATH)
    
    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(f"FAISS index built and saved successfully in {elapsed:.2f}s")
    return vs

# ── 5. Retriever ──────────────────────────────────────────
def get_retriever(vector_store: FAISS, k: int = 5):
    """Create MMR retriever for diverse results."""
    logger.info(f"Initializing retriever with k={k}, MMR search")
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
logger.info("RAG prompt template configured")

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
        logger.info(f"Retrieving context for question: {state['question'][:50]}...")
        docs = retriever.invoke(state["question"])
        logger.info(f"Retrieved {len(docs)} relevant documents")

        parts, sources = [], []
        for i, doc in enumerate(docs, 1):
            src = doc.metadata.get("source", "Unknown")
            src_short = Path(src).name if src != "Unknown" else "Unknown"
            parts.append(f"[{i}] ({src_short})\n{doc.page_content}")
            if src_short not in sources:
                sources.append(src_short)

        logger.info(f"Sources identified: {', '.join(sources)}")    
        return {**state, "context": "\n\n---\n\n".join(parts), "sources": sources}

    def generate_node(state: RAGState) -> RAGState:
        logger.info("Generating answer using LLM...")  
        chain  = rag_prompt | llm | StrOutputParser()
        try:
            answer = chain.invoke({
                "question":     state["question"],
                "context":      state["context"],
                "chat_history": state["messages"][:-1],
            })
            logger.info(f"Answer generated (length: {len(answer)} chars)")
            logger.debug(f"Answer preview: {answer[:100]}...")
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            answer = f"I encountered an error while generating the answer: {str(e)}"                  
        return {**state, "answer": answer, "messages": [AIMessage(content=answer)]}

    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_edge(START,      "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    logger.info("LangGraph workflow built successfully")
    return graph.compile(checkpointer=InMemorySaver())


# ── 9. Public Chatbot Class ───────────────────────────────
class MLRagChatbot:
    def __init__(self, force_rebuild: bool = False):
        logger.info("=" * 60)
        logger.info("Initializing ML RAG Chatbot instance")
        logger.info(f"Force rebuild: {force_rebuild}")
        
        try:
            self.vector_store    = build_vector_store(force_rebuild=force_rebuild)
            self.retriever       = get_retriever(self.vector_store)
            self.graph           = build_rag_graph(self.retriever)
            self._thread_counter = 0
            logger.info("ML RAG Chatbot initialized successfully")
            logger.info("=" * 60)
        except Exception as e:
            logger.error(f"Failed to initialize chatbot: {e}")
            raise

    def new_thread_id(self) -> str:
        self._thread_counter += 1
        thread_id = f"session-{self._thread_counter}"
        logger.info(f"Created new thread: {thread_id}")
        return thread_id

    def chat(self, question: str, thread_id: str) -> dict:
        logger.info(f"Processing question [Thread: {thread_id}]: {question[:100]}...")
        start_time = datetime.now()
        
        config = {"configurable": {"thread_id": thread_id}}
        try: 
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
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(f"Question answered in {elapsed:.2f}s using {len(result['sources'])} sources")
            return {"answer": result["answer"], "sources": result["sources"]}
        except Exception as e:
            logger.error(f"Error processing question: {e}", exc_info=True)
            return {
                "answer": f"Error: {str(e)}",
                "sources": []
            }
