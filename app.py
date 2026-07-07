import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="RAG Chat", page_icon="🔍", layout="centered")

st.title("RAG Chat App")
st.caption("Ask questions about your documents using AI-powered retrieval.")

# --- Sidebar ---
with st.sidebar:
    st.header("Settings")
    topic = st.selectbox("Filter by topic", ["databases", "python", "all"])
    top_k = st.slider("Number of chunks to retrieve", min_value=1, max_value=5, value=2)
    st.divider()
    st.markdown("**Backend:** `http://127.0.0.1:8000`")
    if st.button("Check API Health"):
        try:
            res = requests.get(f"{API_URL}/health", timeout=3)
            data = res.json()
            st.success(f"API is running. {data['chunks_indexed']} chunks indexed.")
        except Exception:
            st.error("API is not reachable. Start the FastAPI server first.")

# --- Chat history ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander("Sources"):
                for src in msg["sources"]:
                    st.markdown(f"> {src}")

# --- Input ---
question = st.chat_input("Ask a question about your documents...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching and generating answer..."):
            try:
                payload = {
                    "question": question,
                    "top_k": top_k
                }
                if topic != "all":
                    payload["topic"] = topic

                res = requests.post(f"{API_URL}/ask", json=payload, timeout=30)

                if res.status_code == 200:
                    data = res.json()
                    answer = data["answer"]
                    sources = data["sources"]

                    st.markdown(answer)
                    with st.expander("Sources"):
                        for src in sources:
                            st.markdown(f"> {src}")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                else:
                    try:
                        error = res.json().get("detail", "Unknown error")
                    except Exception:
                        error = res.text or f"HTTP {res.status_code}"
                    st.error(f"Error: {error}")

            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API. Run: `.venv\\Scripts\\python.exe -m uvicorn src.api:app --port 8000`")
            except Exception as e:
                st.error(f"Unexpected error: {e}")
