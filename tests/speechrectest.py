#!/usr/bin/env python3
"""Voice recognition demo for the capstone (Week 8 slice).

Pipeline: wake word ("hey jarvis") -> record until trailing silence -> Whisper.

Setup on the 1080 Ti workstation, inside the project venv:

    sudo apt install libportaudio2
    uv pip install sounddevice openwakeword webrtcvad-wheels faster-whisper

Design notes:
- openWakeWord and webrtcvad run on CPU, keeping the GPU free for
  transcription and downstream LLM inference.
- faster-whisper uses CTranslate2 (not torch), so it does not disturb the
  pinned torch==2.2.2+cu118 stack. Pascal (SM 6.1) supports DP4A, so
  compute_type="int8" works on the 1080 Ti; the loader falls back to CPU.
- webrtcvad has no wheel for Python 3.11; install webrtcvad-wheels
  (same import name).

Usage:
    python demo_voice.py                  # wake-word mode
    python demo_voice.py --push-to-talk   # press Enter to record instead
    python demo_voice.py --list-devices   # find your USB mic index
    python demo_voice.py --device 3       # use a specific input device
"""

import argparse
import collections
import queue
import sys
import time

import numpy as np

SAMPLE_RATE = 16_000
CHUNK = 1280                 # 80 ms — the frame size openWakeWord expects
VAD_FRAME = 480              # 30 ms — a frame size webrtcvad accepts
BYTES_PER_SAMPLE = 2         # int16 mono

WAKE_THRESHOLD = 0.5
PREROLL_CHUNKS = 5           # ~0.4 s of audio kept from before speech starts
TRAILING_SILENCE_MS = 800    # end of utterance after this much silence
LEADING_TIMEOUT_S = 6.0      # give up if no speech follows the wake word
MAX_UTTERANCE_S = 15.0


def iter_vad_frames(carry: bytes, chunk: bytes):
    """Split carry+chunk into exact 30 ms VAD frames; return (frames, remainder).

    Kept as a pure function so it can be unit-tested without a microphone.
    """
    buf = carry + chunk
    frame_bytes = VAD_FRAME * BYTES_PER_SAMPLE
    n = len(buf) // frame_bytes
    frames = [buf[i * frame_bytes:(i + 1) * frame_bytes] for i in range(n)]
    return frames, buf[n * frame_bytes:]


def load_whisper(model_size: str):
    from faster_whisper import WhisperModel
    for device, compute in (("cuda", "int8"), ("cpu", "int8")):
        try:
            model = WhisperModel(model_size, device=device, compute_type=compute)
            print(f"[whisper] loaded '{model_size}' on {device} ({compute})")
            return model
        except Exception as e:
            print(f"[whisper] {device} unavailable ({e}); trying next", file=sys.stderr)
    raise RuntimeError("could not load faster-whisper on cuda or cpu")


def load_wakeword(name: str):
    import openwakeword
    from openwakeword.model import Model
    openwakeword.utils.download_models([name])  # no-op if already cached
    model = Model(wakeword_models=[name], inference_framework="onnx")
    print(f"[wakeword] listening for '{name.replace('_', ' ')}'")
    return model


def record_utterance(audio_q: "queue.Queue[np.ndarray]", vad, preroll) -> np.ndarray:
    """Consume mic chunks until trailing silence; return int16 mono audio."""
    chunks = list(preroll)
    carry = b""
    silence_ms = 0
    speech_seen = False
    started = time.monotonic()

    while True:
        chunk = audio_q.get()
        chunks.append(chunk)
        frames, carry = iter_vad_frames(carry, chunk.tobytes())
        for f in frames:
            if vad.is_speech(f, SAMPLE_RATE):
                speech_seen = True
                silence_ms = 0
            else:
                silence_ms += 30

        elapsed = time.monotonic() - started
        if speech_seen and silence_ms >= TRAILING_SILENCE_MS:
            break
        if not speech_seen and elapsed > LEADING_TIMEOUT_S:
            print("[record] no speech detected, giving up")
            break
        if elapsed > MAX_UTTERANCE_S:
            print("[record] max utterance length reached")
            break

    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int16)


def transcribe(whisper, audio_int16: np.ndarray) -> str:
    if audio_int16.size == 0:
        return ""
    audio = audio_int16.astype(np.float32) / 32768.0
    t0 = time.monotonic()
    segments, _info = whisper.transcribe(audio, beam_size=1, language="en")
    text = " ".join(s.text.strip() for s in segments).strip()
    print(f"[whisper] transcribed {audio.size / SAMPLE_RATE:.1f}s of audio "
          f"in {time.monotonic() - t0:.2f}s")
    return text


def drain(q: queue.Queue):
    try:
        while True:
            q.get_nowait()
    except queue.Empty:
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="small", help="faster-whisper size (default: small)")
    ap.add_argument("--wakeword", default="hey_jarvis", help="openWakeWord model name")
    ap.add_argument("--push-to-talk", action="store_true", help="Enter key instead of wake word")
    ap.add_argument("--device", type=int, default=None, help="input device index")
    ap.add_argument("--list-devices", action="store_true")
    args = ap.parse_args()

    import sounddevice as sd

    if args.list_devices:
        print(sd.query_devices())
        return

    import webrtcvad
    vad = webrtcvad.Vad(2)  # 0 = permissive .. 3 = aggressive
    whisper = load_whisper(args.model)
    wake = None if args.push_to_talk else load_wakeword(args.wakeword)

    audio_q: "queue.Queue[np.ndarray]" = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print(f"[audio] {status}", file=sys.stderr)
        audio_q.put(indata[:, 0].copy())

    preroll = collections.deque(maxlen=PREROLL_CHUNKS)

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                        blocksize=CHUNK, device=args.device, callback=callback):
        print("[demo] ready. Ctrl+C to quit.")
        while True:
            if args.push_to_talk:
                input("\nPress Enter, then speak: ")
                drain(audio_q)
                preroll.clear()
            else:
                # Wait for the wake word.
                while True:
                    chunk = audio_q.get()
                    preroll.append(chunk)
                    score = wake.predict(chunk)[args.wakeword]
                    if score > WAKE_THRESHOLD:
                        print(f"\n[wakeword] detected (score {score:.2f}) — speak now")
                        wake.reset()
                        break

            audio = record_utterance(audio_q, vad, preroll)
            preroll.clear()
            text = transcribe(whisper, audio)
            print(f'>>> "{text}"' if text else ">>> (nothing transcribed)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[demo] bye")