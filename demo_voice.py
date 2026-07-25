"""Voice front-end demo for PDVA.

Pipeline: mic -> openWakeWord ("jarona") -> webrtcvad end-pointing ->
faster-whisper transcription -> print the recognized question.

This is the capture + STT half of the assistant only. It does not call
RAGPipeline / LocalLLM / Speaker — wiring those in is the Week 10
integration step. Run this standalone to sanity-check the mic, wake
word, and STT before that integration.

Usage:
    python voice_demo.py
    Ctrl+C to quit.
"""
from __future__ import annotations

import collections
import queue
import sys

import numpy as np
import sounddevice as sd
import webrtcvad
from openwakeword.model import Model as WakeWordModel
from faster_whisper import WhisperModel

from pdva import config

# --- Audio constants ------------------------------------------------------
SAMPLE_RATE = 16000

# openWakeWord requires exactly 1280-sample (80 ms) int16 frames.
OWW_FRAME_SAMPLES = 1280

# webrtcvad only accepts 10/20/30 ms frames. We use 30 ms (480 samples).
VAD_FRAME_MS = 30
VAD_FRAME_SAMPLES = SAMPLE_RATE * VAD_FRAME_MS // 1000

WAKE_THRESHOLD = 0.5
# How much trailing silence (in VAD frames) ends an utterance.
SILENCE_FRAMES_TO_STOP = int(0.8 * 1000 / VAD_FRAME_MS)   # ~0.8s of silence
MAX_UTTERANCE_FRAMES = int(15 * 1000 / VAD_FRAME_MS)      # 15s hard cap
VAD_AGGRESSIVENESS = 2


def audio_stream():
    """Yields int16 mono frames of `frame_samples` length from the mic.

    A single sounddevice InputStream feeds a queue; frames of whatever
    size the caller wants are assembled from it via `read_frames`.
    """
    q: queue.Queue[np.ndarray] = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print(f"[audio] {status}", file=sys.stderr)
        q.put(indata.copy())

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        blocksize=VAD_FRAME_SAMPLES,  # smallest granularity we need
        callback=callback,
    )
    return stream, q


def read_frame(q: "queue.Queue[np.ndarray]", n_samples: int, leftover: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pulls exactly n_samples int16 samples, buffering across queue reads.

    Returns (frame, new_leftover).
    """
    buf = leftover
    while buf.shape[0] < n_samples:
        chunk = q.get().reshape(-1)
        buf = np.concatenate([buf, chunk])
    frame, rest = buf[:n_samples], buf[n_samples:]
    return frame, rest


def wait_for_wake_word(oww: WakeWordModel, q: "queue.Queue[np.ndarray]") -> np.ndarray:
    """Blocks until the wake word fires. Returns leftover samples not yet consumed."""
    leftover = np.zeros(0, dtype=np.int16)
    print("Listening for wake word...")
    while True:
        frame, leftover = read_frame(q, OWW_FRAME_SAMPLES, leftover)
        prediction = oww.predict(frame)
        # Custom models are keyed by filename stem in the prediction dict;
        # with a single custom model loaded, take the max score regardless
        # of the exact key rather than assuming a specific name.
        score = max(prediction.values())
        if score >= WAKE_THRESHOLD:
            print(f"Wake word detected (score={score:.2f}).")
            return leftover


def record_utterance(vad: webrtcvad.Vad, q: "queue.Queue[np.ndarray]", leftover: np.ndarray) -> np.ndarray:
    """Records until webrtcvad sees sustained silence, or a max duration is hit.

    Returns the recorded audio as float32 in [-1, 1], ready for faster-whisper.
    """
    print("Recording...")
    frames: list[np.ndarray] = []
    silence_run = 0
    n = 0
    while n < MAX_UTTERANCE_FRAMES:
        frame, leftover = read_frame(q, VAD_FRAME_SAMPLES, leftover)
        frames.append(frame)
        is_speech = vad.is_speech(frame.tobytes(), SAMPLE_RATE)
        silence_run = 0 if is_speech else silence_run + 1
        n += 1
        # Require at least one speech frame before we start counting silence
        # as an end-of-utterance signal, so we don't stop on leading silence.
        if silence_run >= SILENCE_FRAMES_TO_STOP and n > silence_run:
            break
    print("Done recording.")
    audio_i16 = np.concatenate(frames)
    return audio_i16.astype(np.float32) / 32768.0


def main():
    oww = WakeWordModel(wakeword_model_paths=[config.OPENWAKEWORD_MODEL])
    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
    whisper = WhisperModel(
        config.WHISPER_MODEL,
        device=config.WHISPER_DEVICE,
        compute_type=config.WHISPER_COMPUTE,
    )

    stream, q = audio_stream()
    with stream:
        try:
            while True:
                leftover = wait_for_wake_word(oww, q)
                audio = record_utterance(vad, q, leftover)
                segments, _info = whisper.transcribe(audio, language="en")
                text = " ".join(seg.text.strip() for seg in segments)  # consume generator fully before timing/using it
                if text:
                    print(f"You said: {text}")
                else:
                    print("(no speech recognized)")
        except KeyboardInterrupt:
            print("\nExiting.")


if __name__ == "__main__":
    main()