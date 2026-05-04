"""
============================================================
  ML & Deep Learning RAG Chatbot — Streamlit Frontend
  Author: Bivor
============================================================
"""

import streamlit as st
from pathlib import Path

# ── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="ML Guru – RAG Chatbot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

  :root {
    --bg-deep:        #0a0e1a;
    --bg-card:        #111827;
    --bg-input:       #1a2235;
    --accent-cyan:    #00d4ff;
    --accent-blue:    #3b82f6;
    --accent-lime:    #a3e635;
    --text-primary:   #f1f5f9;
    --text-secondary: #94a3b8;
    --border:         rgba(255,255,255,0.08);
    --user-bg:        linear-gradient(135deg, #1e3a5f 0%, #1a2d4a 100%);
    --ai-bg:          linear-gradient(135deg, #0f2a1e 0%, #0d1f17 100%);
    --radius:         14px;
  }

  .stApp { background: var(--bg-deep); font-family: 'Space Grotesk', sans-serif; }
  * { box-sizing: border-box; }
  p, li, span { color: var(--text-primary); }

  [data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-right: 1px solid var(--border);
  }
  [data-testid="stSidebar"] * { color: var(--text-primary) !important; }

  .header-banner {
    background: linear-gradient(135deg, #0a1628 0%, #0d2137 50%, #0a1628 100%);
    border: 1px solid rgba(0,212,255,0.2);
    border-radius: var(--radius);
    padding: 28px 36px; margin-bottom: 24px;
    position: relative; overflow: hidden;
  }
  .header-banner::before {
    content: ''; position: absolute; top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(circle at 30% 40%, rgba(0,212,255,0.06) 0%, transparent 60%);
    pointer-events: none;
  }
  .header-banner h1 {
    font-size: 2rem; font-weight: 700;
    background: linear-gradient(90deg, #00d4ff, #3b82f6, #a3e635);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0 0 6px 0;
  }
  .header-banner p { color: var(--text-secondary); margin: 0; font-size: 0.95rem; }

  .msg-user, .msg-ai {
    border-radius: var(--radius); padding: 18px 22px;
    border: 1px solid var(--border); margin-bottom: 14px;
    animation: fadeUp 0.3s ease;
  }
  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  .msg-user { background: var(--user-bg); border-color: rgba(59,130,246,0.25); }
  .msg-ai   { background: var(--ai-bg);   border-color: rgba(0,212,255,0.2); }

  .msg-label {
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.1em;
    text-transform: uppercase; margin-bottom: 8px;
    font-family: 'JetBrains Mono', monospace;
  }
  .msg-user .msg-label { color: #60a5fa; }
  .msg-ai   .msg-label { color: #00d4ff; }
  .msg-content { font-size: 0.96rem; line-height: 1.7; color: var(--text-primary); white-space: pre-wrap; }

  .sources-row { margin-top: 12px; display: flex; flex-wrap: wrap; gap: 7px; align-items: center; }
  .source-label { font-size: 0.72rem; color: var(--text-secondary); font-weight: 500; margin-right: 4px; }
  .source-pill {
    background: rgba(0,212,255,0.08); border: 1px solid rgba(0,212,255,0.25);
    color: #00d4ff; border-radius: 20px; padding: 3px 11px; font-size: 0.74rem;
    font-family: 'JetBrains Mono', monospace;
  }

  .thinking {
    display: flex; align-items: center; gap: 12px;
    background: var(--ai-bg); border: 1px solid rgba(0,212,255,0.2);
    border-radius: var(--radius); padding: 16px 22px; margin-bottom: 14px;
  }
  .thinking-dots span {
    display: inline-block; width: 8px; height: 8px;
    background: var(--accent-cyan); border-radius: 50%;
    animation: bounce 1.2s infinite;
  }
  .thinking-dots span:nth-child(2) { animation-delay: 0.2s; }
  .thinking-dots span:nth-child(3) { animation-delay: 0.4s; }
  @keyframes bounce {
    0%, 80%, 100% { transform: scale(0.6); opacity: 0.4; }
    40%            { transform: scale(1.0); opacity: 1.0; }
  }
  .thinking-text { color: var(--text-secondary); font-size: 0.9rem; font-style: italic; }

  .stat-card {
    background: rgba(255,255,255,0.04); border: 1px solid var(--border);
    border-radius: 10px; padding: 14px 16px; margin-bottom: 10px; text-align: center;
  }
  .stat-num { font-size: 1.8rem; font-weight: 700; color: var(--accent-cyan); font-family: 'JetBrains Mono', monospace; }
  .stat-lbl { font-size: 0.78rem; color: var(--text-secondary); margin-top: 2px; }

  .loading-status {
    background: var(--bg-card); border: 1px solid rgba(0,212,255,0.2);
    border-radius: var(--radius); padding: 30px; text-align: center; margin: 40px auto;
    max-width: 500px;
  }
  .loading-status h3 { color: var(--accent-cyan); margin-bottom: 8px; }
  .loading-status p  { color: var(--text-secondary); font-size: 0.9rem; margin: 4px 0; }

  .stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #1e3a8a) !important;
    color: #fff !important; border: none !important;
    border-radius: 10px !important; font-weight: 600 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    transition: all 0.2s ease !important; text-align: left !important;
  }
  .stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(59,130,246,0.4) !important;
    background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
  }

  hr { border-color: var(--border) !important; }
  #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Load Chatbot — cached so it only runs ONCE ────────────
# @st.cache_resource ensures this function is called only once
# across all browser sessions. Subsequent loads are instant.
@st.cache_resource(show_spinner=False)
def load_chatbot():
    """
    Loads the full RAG pipeline once and caches it.
    - First run : downloads embeddings + builds FAISS index (~1-3 min)
    - All later runs: loads FAISS from disk (~3-5 seconds)
    """
    from rag_backend import MLRagChatbot
    return MLRagChatbot(force_rebuild=False)

# ── Boot Sequence ─────────────────────────────────────────
# Show a friendly loading screen instead of a blocking spinner
if "chatbot" not in st.session_state:
    # Display loading UI immediately while cache_resource runs
    load_placeholder = st.empty()

    with load_placeholder.container():
        st.markdown("""
        <div class='loading-status'>
          <div style='font-size:3rem;margin-bottom:12px;'>🧠</div>
          <h3>ML Guru is starting up…</h3>
          <p>Loading embedding model &amp; FAISS index.</p>
          <p style='color:#475569;font-size:0.82rem;margin-top:12px;'>
            ⏱️ First run: ~1–3 min (downloads model + builds index)<br>
            ⚡ After that: always loads in under 5 seconds
          </p>
        </div>
        """, unsafe_allow_html=True)

    # This is the actual blocking call — runs in background via cache
    chatbot = load_chatbot()

    # Store in session state once ready
    st.session_state.chatbot        = chatbot
    st.session_state.thread_id      = chatbot.new_thread_id()
    st.session_state.messages       = []
    st.session_state.total_questions = 0
    st.session_state.sources_seen   = set()
    st.session_state.quick_input    = ""

    load_placeholder.empty()  # Clear loading screen
    st.rerun()                # Rerun to show the actual chat UI

# ── Ensure session state keys exist ──────────────────────
for key, default in [
    ("messages",        []),
    ("total_questions", 0),
    ("sources_seen",    set()),
    ("quick_input",     ""),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 ML Guru")
    st.markdown(
        "<p style='color:#94a3b8;font-size:0.85rem;line-height:1.6;'>"
        "Your personal ML &amp; Deep Learning RAG assistant — "
        "answers come from <strong>your own study materials</strong>.</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown("### 📊 Session Stats")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f"<div class='stat-card'><div class='stat-num'>{st.session_state.total_questions}</div>"
            f"<div class='stat-lbl'>Questions</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(
            f"<div class='stat-card'><div class='stat-num'>{len(st.session_state.sources_seen)}</div>"
            f"<div class='stat-lbl'>Sources Used</div></div>", unsafe_allow_html=True)

    st.divider()
    st.markdown("### ⚡ Quick Topics")

    quick_topics = [
        "What is RAG and how does it work?",
        "Explain supervised vs unsupervised learning",
        "How does LangGraph differ from LangChain?",
        "What are the 4 steps to build a RAG system?",
        "What is fine-tuning and when should I use it?",
        "Explain FAISS vs Pinecone for vector stores",
        "What are the types of neural network architectures?",
        "How does MMR retrieval improve RAG quality?",
        "What is in-context learning?",
        "Explain the difference between RAG and fine-tuning",
    ]
    for topic in quick_topics:
        label = f"→ {topic[:44]}…" if len(topic) > 44 else f"→ {topic}"
        if st.button(label, key=f"qt_{topic[:20]}", use_container_width=True):
            st.session_state.quick_input = topic
            st.rerun()

    st.divider()
    st.markdown("### ⚙️ Controls")
    if st.button("🗑️  Clear Chat History", use_container_width=True):
        st.session_state.messages        = []
        st.session_state.total_questions = 0
        st.session_state.sources_seen    = set()
        st.session_state.thread_id       = st.session_state.chatbot.new_thread_id()
        st.rerun()

    if st.button("🔄  Rebuild FAISS Index", use_container_width=True):
        st.cache_resource.clear()
        from rag_backend import MLRagChatbot
        with st.spinner("Rebuilding index from your ML files…"):
            st.session_state.chatbot   = MLRagChatbot(force_rebuild=True)
            st.session_state.thread_id = st.session_state.chatbot.new_thread_id()
        st.success("✅ Index rebuilt!")
        st.rerun()

    st.divider()
    st.markdown(
        "<p style='font-size:0.76rem;color:#475569;text-align:center;line-height:1.7;'>"
        "🔗 LangChain · LangGraph · FAISS<br>"
        "🤖 Groq LLaMA 3.3 · 70B Versatile<br>"
        "🗂️ Your Desktop Study Materials"
        "</p>", unsafe_allow_html=True)

# ── Main Area ─────────────────────────────────────────────
st.markdown("""
<div class='header-banner'>
  <h1>🧠 ML Guru — RAG Chatbot</h1>
  <p>Ask anything about Machine Learning, Deep Learning, RAG, LangGraph, and more.<br>
     Every answer is grounded in <strong>your personal study materials</strong> from your Desktop.</p>
</div>
""", unsafe_allow_html=True)

# ── Render One Message ────────────────────────────────────
def render_message(role: str, content: str, sources: list = None):
    if role == "user":
        st.markdown(
            f"<div class='msg-user'><div class='msg-label'>👤 You</div>"
            f"<div class='msg-content'>{content}</div></div>",
            unsafe_allow_html=True)
    else:
        sources_html = ""
        if sources:
            pills = "".join(f"<span class='source-pill'>{s}</span>" for s in sources)
            sources_html = f"<div class='sources-row'><span class='source-label'>📎 Sources:</span>{pills}</div>"
        st.markdown(
            f"<div class='msg-ai'><div class='msg-label'>🤖 ML Guru</div>"
            f"<div class='msg-content'>{content}</div>{sources_html}</div>",
            unsafe_allow_html=True)

# ── Chat History ──────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
    <div style='text-align:center;padding:70px 20px;color:#475569;'>
      <div style='font-size:3.5rem;margin-bottom:18px;'>🎓</div>
      <p style='font-size:1.1rem;font-weight:600;color:#64748b;margin-bottom:8px;'>
        Ready to learn from your own study materials!
      </p>
      <p style='font-size:0.88rem;color:#475569;'>
        Type a question below, or pick a Quick Topic from the sidebar →
      </p>
    </div>
    """, unsafe_allow_html=True)
else:
    for msg in st.session_state.messages:
        render_message(msg["role"], msg["content"], msg.get("sources"))


# ── Input: Chat Box or Sidebar Quick Topic ────────────────
pending = st.session_state.pop("quick_input", "")
user_input = st.chat_input("Ask about ML, Deep Learning, RAG, LangGraph…") or pending

if user_input and user_input.strip():
    question = user_input.strip()

    st.session_state.messages.append({"role": "user", "content": question})
    render_message("user", question)

    placeholder = st.empty()
    placeholder.markdown(
        "<div class='thinking'>"
        "<div class='thinking-dots'><span></span><span></span><span></span></div>"
        "<div class='thinking-text'>Searching your study materials…</div>"
        "</div>", unsafe_allow_html=True)

    try:
        result  = st.session_state.chatbot.chat(question=question, thread_id=st.session_state.thread_id)
        answer  = result["answer"]
        sources = result["sources"]
    except Exception as e:
        answer  = f"⚠️ Error: {str(e)}\n\nCheck that GROQ_API_KEY is set in your .env file."
        sources = []

    placeholder.empty()

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
    st.session_state.total_questions += 1
    st.session_state.sources_seen.update(sources)

    render_message("assistant", answer, sources)
    st.rerun()
