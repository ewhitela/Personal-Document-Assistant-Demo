"""
Jetson-side TTS service (Week 11 edge deployment).

Runs on the Jetson Nano (JetPack 4.6 / Python 3.6). Wraps the `piper`
CLI binary in a tiny Flask app with two endpoints:

  GET  /health      -> {"status": "ok"} if piper binary + voice are usable
  POST /synthesize   -> body {"text": "..."} returns audio/wav bytes

Deliberately shells out to the `piper` binary rather than importing a
Python piper package: JetPack 4.6 ships Python 3.6, and modern piper
python bindings assume newer interpreters. The CLI has ARM builds and
has no such constraint.

Config is read from environment variables so the same file works
unmodified across Jetson units:

  PIPER_BIN     path to the piper executable (default: "piper")
  PIPER_VOICE   path to the .onnx voice model (required)
  PIPER_CONFIG  path to the voice's .onnx.json config (default: voice + ".json")
  PORT          port to listen on (default: 5000)
"""

import os
import subprocess
import tempfile

from flask import Flask, request, send_file, jsonify

app = Flask(__name__)

PIPER_BIN = os.environ.get("PIPER_BIN", "piper")
PIPER_VOICE = os.environ.get("PIPER_VOICE")
PIPER_CONFIG = os.environ.get("PIPER_CONFIG", (PIPER_VOICE + ".json") if PIPER_VOICE else None)
SYNTH_TIMEOUT_S = float(os.environ.get("PIPER_TIMEOUT_S", "25"))


def _piper_ready():
    if not PIPER_VOICE or not os.path.exists(PIPER_VOICE):
        return False, "voice model not found: {}".format(PIPER_VOICE)
    try:
        subprocess.run(
            [PIPER_BIN, "--help"],
            capture_output=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        return False, "piper binary not runnable: {}".format(e)
    return True, "ok"


@app.route("/health", methods=["GET"])
def health():
    ok, detail = _piper_ready()
    return jsonify({"status": "ok" if ok else "error", "detail": detail}), (200 if ok else 503)


@app.route("/synthesize", methods=["POST"])
def synthesize():
    payload = request.get_json(silent=True) or {}
    text = payload.get("text", "").strip()
    if not text:
        return jsonify({"error": "missing 'text' field"}), 400

    ok, detail = _piper_ready()
    if not ok:
        return jsonify({"error": detail}), 503

    out_path = tempfile.mktemp(suffix=".wav")
    cmd = [PIPER_BIN, "--model", PIPER_VOICE, "--output_file", out_path]
    if PIPER_CONFIG and os.path.exists(PIPER_CONFIG):
        cmd += ["--config", PIPER_CONFIG]

    try:
        subprocess.run(
            cmd,
            input=text.encode("utf-8"),
            capture_output=True,
            timeout=SYNTH_TIMEOUT_S,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        return jsonify({"error": "piper failed", "stderr": e.stderr.decode("utf-8", "ignore")}), 500
    except subprocess.TimeoutExpired:
        return jsonify({"error": "piper timed out after {}s".format(SYNTH_TIMEOUT_S)}), 504

    if not os.path.exists(out_path):
        return jsonify({"error": "piper produced no output"}), 500

    return send_file(out_path, mimetype="audio/wav", as_attachment=False)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)