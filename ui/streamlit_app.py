"""Streamlit chat UI for the CAG API.

This is a thin client: it holds no LLM/provider logic and only talks to the
FastAPI backend over HTTP. Run the API first, then:

    streamlit run ui/streamlit_app.py
"""

import os

import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
REQUEST_TIMEOUT_SECONDS = 120

st.set_page_config(page_title="HR Policy Assistant (CAG)", page_icon="📄")


def api_get(path: str) -> dict:
    response = requests.get(f"{API_BASE_URL}{path}", timeout=10)
    response.raise_for_status()
    return response.json()


def api_chat(question: str) -> dict:
    response = requests.post(
        f"{API_BASE_URL}/api/v1/chat",
        json={"question": question},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    if not response.ok:
        detail = response.json().get("detail", response.text) if response.content else response.reason
        raise RuntimeError(f"{response.status_code}: {detail}")
    return response.json()


def render_stats(stats: dict) -> None:
    st.caption(
        f"⏱ {stats['latency_seconds']:.2f}s · "
        f"input {stats['input_tokens']} · "
        f"cached {stats['cached_tokens']} · "
        f"total {stats['total_tokens']} tokens"
    )


# ---- Sidebar: backend status + session totals ---------------------------------
with st.sidebar:
    st.header("Backend")
    try:
        health = api_get("/api/v1/health")
        st.success(f"API {health['status']}")
        st.markdown(
            f"**Provider:** {health['provider']}  \n"
            f"**Model:** {health['model']}  \n"
            f"**Cache mode:** {health['cache_mode']}  \n"
            f"**Document:** {health['document']} (~{health['document_words']} words)"
        )
        totals = api_get("/api/v1/stats")
        st.header("Session totals")
        col1, col2 = st.columns(2)
        col1.metric("Questions", totals["questions_asked"])
        col2.metric("Cached tokens", totals["cached_tokens_reused"])
        st.metric("Total tokens billed", totals["total_tokens_billed"])
    except requests.RequestException as exc:
        st.error(f"Cannot reach API at {API_BASE_URL}\n\n{exc}")

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ---- Chat ---------------------------------------------------------------------
st.title("📄 HR Leave Policy Assistant")
st.caption("Answers come from the cached policy document via Cache-Augmented Generation.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("stats"):
            render_stats(message["stats"])

if question := st.chat_input("Ask about the leave policy..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                reply = api_chat(question)
                answer, stats = reply["answer"], reply["stats"]
            except (requests.RequestException, RuntimeError) as exc:
                answer, stats = f"⚠️ Request failed: {exc}", None
        st.markdown(answer)
        if stats:
            render_stats(stats)

    st.session_state.messages.append({"role": "assistant", "content": answer, "stats": stats})
    st.rerun()  # refresh sidebar session totals
