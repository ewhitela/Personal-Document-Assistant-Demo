"""Week 10: Streamlit UI for the PDVA service.

Thin HTTP client of service/app.py, no model code here. Start the service
first, then:

    streamlit run ui/streamlit_app.py
"""
from __future__ import annotations

import base64
import os

import requests
import streamlit as st

API_URL = os.environ.get("PDVA_API_URL", "http://127.0.0.1:8080")

st.set_page_config(page_title="Personal Document Assistant", page_icon="🗂️")
st.title("Personal Document Assistant")


def api(method: str, path: str, **kwargs):
    """Call the service; return parsed JSON or raise with a readable message."""
    r = requests.request(method, f"{API_URL}{path}", timeout=120, **kwargs)

    if not r.ok:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        raise RuntimeError(f"{method} {path} -> {r.status_code}: {detail}")
    
    return r


def show_result(body: dict) -> None:
    """Render transcript / answer / audio / sources / timings from a response."""
    if body.get("transcript"):
        st.markdown(f"**You asked:** {body['transcript']}")

    if not body.get("answer"):
        st.warning("No speech detected — try again.")
        return
    
    st.markdown(body["answer"])

    if body.get("audio_b64"):
        st.audio(base64.b64decode(body["audio_b64"]), format="audio/wav",
                 autoplay=True)
        
    sources = body.get("sources", [])

    if sources:
        with st.expander(f"Sources ({len(sources)} passages)"):
            for i, p in enumerate(sources, 1):
                st.markdown(f"**[{i}] {p['source']}** — score {p['score']:.3f}")
                st.caption(p["text"])

    timings = body.get("timings", {})

    if timings:
        parts = [f"{k[:-2]} {v:.2f}s" for k, v in timings.items() if k != "total_s"]
        total = timings.get("total_s", 0.0)
        over = " ⚠️ over 3s budget" if total > 3.0 else ""
        st.caption(" · ".join(parts) + f" · **total {total:.2f}s**{over}")

with st.sidebar:
    st.subheader("Status")

    try:
        h = api("GET", "/health").json()

        st.write(
            f"LLM {'✅' if h['llm_ready'] else '❌'} · "
            f"STT {'✅' if h['stt_ready'] else '❌'} · "
            f"TTS {'✅' if h['tts_ready'] else '❌'}"
        )
        
        st.caption(f"{h['indexed_chunks']} chunks indexed")
        service_up = True
    except Exception as e:
        st.error(f"Service unreachable at {API_URL}\n\n{e}")
        service_up = False

    if service_up:
        st.subheader("Documents")
        docs = api("GET", "/documents").json()["documents"]

        for name in docs:
            col1, col2 = st.columns([5, 1])
            col1.write(name)
            
            if col2.button("✕", key=f"del_{name}", help=f"Remove {name}"):
                with st.spinner(f"Removing {name} and rebuilding index…"):
                    api("DELETE", f"/documents/{name}")
                
                st.rerun()

        uploads = st.file_uploader("Add documents", type=["txt", "md", "pdf"],
                                   accept_multiple_files=True)
        
        if uploads and st.button("Index uploads", type="primary"):
            files = [("files", (f.name, f.getvalue())) for f in uploads]

            with st.spinner("Indexing…"):
                res = api("POST", "/documents", files=files).json()
            
            st.success(f"Indexed {res['chunks_added']} chunks "
                       f"in {res['index_s']:.1f}s")
            
            st.rerun()

        if docs and st.button("Clear all documents"):
            api("DELETE", "/documents")
            st.rerun()

    speak_answers = st.toggle("Speak answers", value=False, help="Return a Piper-synthesized WAV with each answer")


# Main: ask by voice or text

if not service_up:
    st.stop()

voice_tab, text_tab = st.tabs(["🎤 Voice", "⌨️ Text"])

with voice_tab:
    recording = st.audio_input("Record your question")

    if recording is not None and st.button("Ask", key="ask_voice", type="primary"):
        with st.spinner("Transcribing and answering…"):
            body = api(
                "POST", "/voice/ask",
                params={"speak": str(speak_answers).lower()},
                files={"audio": ("question.wav", recording.getvalue(), "audio/wav")},
            ).json()

        show_result(body)

with text_tab:
    question = st.text_input("Your question")

    if st.button("Ask", key="ask_text", type="primary") and question.strip():
        with st.spinner("Answering…"):
            body = api("POST", "/ask", json={"question": question, "speak": speak_answers}).json()
            
        show_result(body)
