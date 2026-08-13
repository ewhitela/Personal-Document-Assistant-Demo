"""Warm the local cache for every model the assistant loads at startup.
Run this once while online; after that, HF_HUB_OFFLINE=1 covers cold starts.
"""

from faster_whisper import WhisperModel
from openwakeword.utils import download_models
from sentence_transformers import SentenceTransformer

SentenceTransformer("all-MiniLM-L6-v2")
WhisperModel("base.en", device="cpu", compute_type="int8")
download_models()  # fetches oww's base melspectrogram/embedding onnx files

print("Prefetch complete.")
