import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pdva.tts import split_sentences, Speaker  # noqa: E402

sp = Speaker()


sp.say("        Hint: self.voice = PiperVoice.load(voice_path, use_cuda=use_cuda). The voice_path points at a downloaded .onnx file; its .onnx.json config must sit next to it. use_cuda=True needs the onnxruntime-gpu package.")