

"""Week 8: Text-to-speech via Piper.

You implement every function and method marked `raise NotImplementedError`.
Keep the signatures stable: the Week 10 service calls synthesize (or say) to
speak the RAG answer aloud.

Before you start:
    pip install piper-tts
    python -m piper.download_voices en_US-lessac-medium   # fetches the .onnx voice
    (playback, for the build path: pip install sounddevice soundfile)

Libraries you will import when you implement this:
    import wave
    from piper import PiperVoice

Note on the package: the maintained Piper is OHF-Voice/piper1-gpl and its current
method is voice.synthesize_wav(text, wav_file). Older examples use
voice.synthesize(...); check which your installed version exposes. Piper is
licensed GPL-3.0, which matters if this is ever redistributed.

Run `python tests/test_week8_tts.py` after implementing. The integration check
skips cleanly if Piper or the voice file is missing.
"""
from __future__ import annotations

import logging
import re
import tempfile
import os
import wave

try:
    from piper import PiperVoice
except ImportError:
    PiperVoice = None

import soundfile as sf
import sounddevice as sd

from . import config

logger = logging.getLogger(__name__)

def split_sentences(text: str) -> list[str]:
    """Split text into sentence-sized pieces for smoother synthesis.

    Args:
        text: the answer text to speak.

    Returns:
        A list of non-empty, stripped sentence strings.

    Behavior:
        - Split on sentence-ending punctuation (. ! ?), keeping the pieces.
        - Drop empty pieces and surrounding whitespace.
        - Piper prosody is better on several short sentences than on one long
          block, and short pieces let playback start sooner.
    """

    pieces = re.split(r'([.!?]+)', text)

    sentences = []
    for i in range(0, len(pieces) - 1, 2):
        sentence = (pieces[i] + pieces[i + 1]).strip()
        if sentence:
            sentences.append(sentence)

    if len(pieces) % 2 == 1:
        tail = pieces[-1].strip()
        if tail:
            sentences.append(tail)

    return sentences

class Speaker:
    """Wraps a Piper voice and turns text into speech."""

    def __init__(self, voice_path: str = config.PIPER_VOICE,
                 use_cuda: bool = config.PIPER_USE_CUDA) -> None:
        """Load the voice model.

        Hint: self.voice = PiperVoice.load(voice_path, use_cuda=use_cuda). The
        voice_path points at a downloaded .onnx file; its .onnx.json config must
        sit next to it. use_cuda=True needs the onnxruntime-gpu package.
        """

        self.voice_path = voice_path
        self.use_cuda = use_cuda
        self.voice = None

        if PiperVoice is None:
            logger.warning("piper-tts not installed; Speaker disabled")
            return
        try:
            self.voice = PiperVoice.load(voice_path, use_cuda=use_cuda)
        except Exception:
            logger.exception("Failed to load Piper voice from %s", voice_path)
            self.voice = None

    def is_ready(self) -> bool:
        """Return True if the voice model is loaded and the file exists.

        Return False (do not raise) if Piper is missing or the voice file is not
        found, so callers can fall back to on-screen text.
        """
        
        return self.voice is not None

    def synthesize(self, text: str, out_path: str) -> str:
        """Synthesize text to a WAV file and return its path.

        Behavior:
            Open out_path as a wave file for writing and call
            self.voice.synthesize_wav(text, wav_file). Return out_path so callers
            can play or store it.
        """

        if not self.is_ready():
            raise RuntimeError("Speaker voice is not loaded")
        
        with wave.open(out_path, "wb") as wav_file:
            self.voice.synthesize_wav(text, wav_file)

        return out_path
    
    def say(self, text: str) -> None:
        """Synthesize text and play it through the speakers.

        Behavior:
            Synthesize to a temporary WAV, then play it with an audio backend
            (for example sounddevice + soundfile). This is the convenience path
            the voice loop uses; synthesize is the testable core it builds on.
        """

        if not self.is_ready():
            raise RuntimeError("Speaker voice is not loaded")
        
        if not text.strip():
            return

        sentences = split_sentences(text) or [text.strip()]

        for sentence in sentences:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                out_path = tmp.name
            try:
                self.synthesize(sentence, out_path)
                data, sr = sf.read(out_path, dtype="float32")
                sd.play(data, sr)
                sd.wait()
            finally:
                os.remove(out_path)

