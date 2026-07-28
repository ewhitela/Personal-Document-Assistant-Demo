"""Week 10: Streamlit UI for the PDVA service.

Thin HTTP client of service/app.py, no model code here. Start the service
first, then:

    streamlit run ui/streamlit_app.py
"""

from __future__ import annotations

import base64
import logging
import os
import time

import requests
import streamlit as st

API_URL = os.environ.get("PDVA_API_URL", "http://127.0.0.1:8080")

logger = logging.getLogger("pdva.ui")

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
        st.audio(base64.b64decode(body["audio_b64"]), format="audio/wav", autoplay=True)

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


def stop_wake_listener_if_running() -> None:
    """Best-effort stop; used when navigating away and on script rerun cleanup."""
    try:
        api("POST", "/voice/wake/stop")
    except Exception as e:
        logger.warning("Could not stop wake listener on view switch: %s", e)


with st.sidebar:
    st.subheader("Status")

    try:
        h = api("GET", "/health").json()

        st.write(
            f"LLM {'✅' if h['llm_ready'] else '❌'} · "
            f"STT {'✅' if h['stt_ready'] else '❌'} · "
            f"TTS {'✅' if h['tts_ready'] else '❌'} · "
            f"Vision {'✅' if h.get('vision_ready') else '❌'}"
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

        uploads = st.file_uploader(
            "Add documents", type=["txt", "md", "pdf"], accept_multiple_files=True
        )

        if uploads and st.button("Index uploads", type="primary"):
            files = [("files", (f.name, f.getvalue())) for f in uploads]

            with st.spinner("Indexing…"):
                res = api("POST", "/documents", files=files).json()

            st.success(f"Indexed {res['chunks_added']} chunks in {res['index_s']:.1f}s")

            st.rerun()

        if docs and st.button("Clear all documents"):
            api("DELETE", "/documents")
            st.rerun()

    speak_answers = st.toggle(
        "Speak answers",
        value=False,
        help="Return a Piper-synthesized WAV with each answer",
    )


# Main: ask by voice or text

if not service_up:
    st.stop()

VIEWS = ["🎤 Voice", "🗣️ Wake word", "⌨️ Text", "🖼️ Image"]
WAKE_VIEW = "🗣️ Wake word"

if "active_view" not in st.session_state:
    st.session_state.active_view = VIEWS[0]

selected_view = st.segmented_control(
    "View", VIEWS, default=st.session_state.active_view, key="view_selector"
)

if selected_view is None:
    # segmented_control allows deselection; treat that as "stay put" rather
    # than losing the previous view entirely.
    selected_view = st.session_state.active_view

# The one thing this whole restructure exists for: if the previous rerun had
# the Wake word view active and this rerun doesn't, the listener's background
# thread (and its GPU-contending onnxruntime session) has no reason to keep
# running. Stop it here, before rendering whatever view we switched to --
# not after, so a slow /ask on the new view never overlaps with it.
if st.session_state.active_view == WAKE_VIEW and selected_view != WAKE_VIEW:
    stop_wake_listener_if_running()

st.session_state.active_view = selected_view

if selected_view == "🎤 Voice":
    recording = st.audio_input("Record your question")

    if recording is not None and st.button("Ask", key="ask_voice", type="primary"):
        with st.spinner("Transcribing and answering…"):
            body = api(
                "POST",
                "/voice/ask",
                params={"speak": str(speak_answers).lower()},
                files={"audio": ("question.wav", recording.getvalue(), "audio/wav")},
            ).json()

        show_result(body)

elif selected_view == WAKE_VIEW:
    st.caption(
        'Say "jarona", then ask your question. This listens on the '
        "microphone attached to the machine running the service — not "
        "your browser's mic — so it only works when the UI and service "
        "share a machine, as in this demo setup. Switching to another view "
        "automatically stops listening."
    )

    wake = api("GET", "/voice/wake/status").json()

    state_labels = {
        "listening": "👂 Listening for the wake word…",
        "recording": "🔴 Recording your question…",
        "answering": "🤔 Thinking…",
        "idle": "Stopped",
    }

    col1, col2 = st.columns([1, 3])

    score_suffix = (
        f" · score {wake['score']:.2f}" if wake.get("score") is not None else ""
    )

    if not wake["running"]:
        if col1.button("Start listening", type="primary", key="wake_start"):
            api(
                "POST",
                "/voice/wake/start",
                params={"speak": str(speak_answers).lower()},
            )
            st.rerun()
        col2.caption(state_labels.get(wake["state"], wake["state"]))
    else:
        if col1.button("Stop listening", key="wake_stop"):
            api("POST", "/voice/wake/stop")
            st.rerun()
        col2.caption(
            state_labels.get(wake["state"], f"⚠️ {wake['state']}") + score_suffix
        )

    # Always check for a pending result, even if the listener thread has
    # already exited by this rerun -- _run() puts the answer on the queue
    # and *then* flips state to idle, so gating this fetch on wake["running"]
    # can miss a result that arrived in exactly that gap (confirmed via
    # manual /voice/wake/result: the answer was sitting there, ready, after
    # the UI had already shown "Stopped").
    result = api("GET", "/voice/wake/result").json()

    if result.get("ready"):
        show_result(result)

    if wake["running"]:
        # Poll once a second while the listener is active. Since only the
        # selected view's code runs at all now (unlike st.tabs, which ran
        # every tab's body every rerun), this no longer fires while another
        # view is showing. Once the listener has finished (one utterance
        # per activation), stop auto-rerunning -- the result above was
        # already fetched this pass if one was ready.
        time.sleep(1)
        st.rerun()

elif selected_view == "⌨️ Text":
    question = st.text_input("Your question")

    if st.button("Ask", key="ask_text", type="primary") and question.strip():
        with st.spinner("Answering…"):
            body = api(
                "POST", "/ask", json={"question": question, "speak": speak_answers}
            ).json()

        show_result(body)

elif selected_view == "🖼️ Image":
    img = st.file_uploader(
        "Image", type=["png", "jpg", "jpeg", "webp"], key="vision_img"
    )
    vq = st.text_input("Question about the image (blank = describe)", key="vision_q")

    if img is not None and st.button("Ask", key="ask_vision", type="primary"):
        st.image(img)
        with st.spinner("Looking…"):
            body = api(
                "POST",
                "/vision/ask",
                params={"speak": str(speak_answers).lower()},
                data={"question": vq},
                files={"image": (img.name, img.getvalue(), img.type)},
            ).json()
        show_result(body)
