"""Week 7: Speech-to-text via faster-whisper.

You implement every function and method marked `raise NotImplementedError`.
Keep the signatures stable: the Week 10 service calls transcribe to turn a
recorded question into text before it reaches the RAG pipeline.

Before you start:
    pip install faster-whisper
    (audio capture, for the build path: pip install sounddevice soundfile)

Libraries you will import when you implement this:
    from faster_whisper import WhisperModel

Run `python tests/test_week7_stt.py` after implementing. The integration check
skips cleanly if the model is not available, so the file is always safe to run.
"""
from __future__ import annotations
from faster_whisper import WhisperModel

from .types import TranscriptSegment
from . import config


def join_segments(segments) -> str:
    """Join transcript segments into one clean string.

    Args:
        segments: an iterable of objects that each have a `.text` attribute
                  (this is what faster-whisper yields).

    Returns:
        The concatenated text, collapsed to single spaces and stripped.

    Behavior:
        - Concatenate each segment's text in order.
        - faster-whisper often prefixes a leading space on each segment; collapse
          runs of whitespace so the result reads naturally.
        - An empty iterable returns an empty string.
    """

    text = " ".join(seg.text for seg in segments)

    return " ".join(text.split())

class Transcriber:
    """Wraps a faster-whisper model and turns audio files into text."""

    def __init__(self, model_size: str = config.WHISPER_MODEL,
                 device: str = config.WHISPER_DEVICE,
                 compute_type: str = config.WHISPER_COMPUTE) -> None:
        """Load the model.

        Hint: self.model = WhisperModel(model_size, device=device,
        compute_type=compute_type). The first load downloads the weights. On the
        1080Ti use device="cuda" and compute_type="float16"; on CPU use "int8".
        Loading can fail if faster-whisper is not installed; let that surface here.
        """

        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.model = None
        
        if WhisperModel is None: 
            return
        
        try:
            self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        except Exception:
            self.model = None

    def is_ready(self) -> bool:
        """Return True if a model is loaded and usable.

        Return False (do not raise) if faster-whisper is missing or the model
        failed to load, so the Week 10 service can degrade gracefully.
        """

        return self.model is not None

    def transcribe(self, audio_path: str) -> str:
        """Transcribe an audio file to a single text string.

        Behavior:
            model.transcribe(audio_path) returns (segments, info) where segments
            is a generator. Pass it through join_segments to get the transcript.
            Accept common formats (wav, mp3, m4a, flac); faster-whisper decodes
            them via its bundled ffmpeg backend.
        """

        if not self.is_ready():
            raise RuntimeError("Transcriber model is not loaded")
        
        segments, _info = self.model.transcribe(audio_path,beam_size=5)
        
        joined_s = join_segments(segments)

        print(joined_s)

        return joined_s

    def transcribe_segments(self, audio_path: str) -> list[TranscriptSegment]:
        """Transcribe and return timed segments.

        Returns:
            A list of TranscriptSegment(start, end, text), one per whisper
            segment. Useful for debugging alignment or building subtitles.
        """

        if not self.is_ready():
            raise RuntimeError("Transcriber model is not loaded")
        
        segments, _info = self.model.transcribe(audio_path,beam_size=5)

        return [TranscriptSegment(start=s.start, end=s.end, text=s.text) for s in segments]
