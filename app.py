"""
app.py — Streamlit chat app for the hospital policy RAG assistant.

Flow:
    user question -> embed -> FAISS similarity search -> top-k chunks
    -> build context -> Groq chat completion -> answer + sources shown

Requires:
    - A "faiss_index" folder in the same directory (built by ingest.py)
    - A Groq API key set in Streamlit secrets as GROQ_API_KEY
      (.streamlit/secrets.toml, or the Secrets panel on Streamlit Cloud)
      -- never typed into the app UI, never shown on screen.

Run locally:
    streamlit run app.py
"""

import streamlit as st

from langchain_community.vectorstores import FAISS

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

from groq import Groq

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
INDEX_DIR = "faiss_index"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-120b"  # adjust here if Groq renames/retires the model
TOP_K = 4

SYSTEM_PROMPT = (
    "You are a hospital policy assistant. Answer the user's question using ONLY "
    "the provided context excerpts from hospital policy documents. If the answer "
    "isn't in the context, say plainly that the policy documents don't cover it — "
    "do not guess or use outside knowledge. Be concise. When you use an excerpt, "
    "refer to it inline by its number, e.g. [1] or [2][3]."
)

st.set_page_config(page_title="Hospital Policy Assistant", page_icon="🏥", layout="centered")

# ---------------------------------------------------------------------------
# API key — read from secrets only, never rendered in a text box
# ---------------------------------------------------------------------------
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error(
        "GROQ_API_KEY not found. Add it to `.streamlit/secrets.toml` "
        "(locally) or the app's Secrets panel (Streamlit Cloud):\n\n"
        '```toml\nGROQ_API_KEY = "your_key_here"\n```'
    )
    st.stop()

client = Groq(api_key=GROQ_API_KEY)


# ---------------------------------------------------------------------------
# Load the FAISS index once and cache it across reruns
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading knowledge base...")
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)


try:
    db = load_vectorstore()
except Exception as e:
    st.error(f"Could not load FAISS index from '{INDEX_DIR}': {e}")
    st.stop()


# ---------------------------------------------------------------------------
# Retrieval + generation
# ---------------------------------------------------------------------------
def retrieve(query: str, k: int = TOP_K):
    return db.similarity_search(query, k=k)


def build_context(chunks) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        dept = c.metadata.get("department", "unknown")
        src = c.metadata.get("source", "unknown")
        parts.append(f"[{i}] (Department: {dept} | Source: {src})\n{c.page_content}")
    return "\n\n".join(parts)


def ask_groq(question: str, context: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.2,
    )
    return response.choices[0].message.content


def unique_sources(chunks):
    seen = set()
    sources = []
    for c in chunks:
        dept = c.metadata.get("department", "unknown")
        src = c.metadata.get("source", "unknown")
        key = (dept, src)
        if key not in seen:
            seen.add(key)
            sources.append({"department": dept, "source": src})
    return sources


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🏥 Hospital Policy Assistant")
st.caption("Ask a question about hospital policy. Answers are grounded in the uploaded policy documents.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Replay chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- **{s['department']}** — `{s['source']}`")

question = st.chat_input("Ask about a hospital policy...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching policies..."):
            chunks = retrieve(question)
            context = build_context(chunks)
            try:
                answer = ask_groq(question, context)
            except Exception as e:
                answer = f"Sorry, the request to Groq failed: {e}"

        st.markdown(answer)

        sources = unique_sources(chunks)
        if sources:
            with st.expander("Sources"):
                for s in sources:
                    st.markdown(f"- **{s['department']}** — `{s['source']}`")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
