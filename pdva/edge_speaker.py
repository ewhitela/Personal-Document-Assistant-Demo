"""
Workstation-side client for the Jetson-hosted Piper TTS service.

New file (Week 11, optional edge deployment) — does not modify the
existing `pdva.tts.Speaker`. EdgeSpeaker exposes the same two methods
(`is_ready`, `synthesize`) with the same signature, so it can be
swapped in for the local `Speaker` in the orchestrator with no other
code changes:

    from pdva.tts import Speaker
    from pdva.edge_speaker import EdgeSpeaker

    speaker = Speaker(...)                       # local
    speaker = EdgeSpeaker("http://jetson.local:5000")  # edge, same interface
"""

import requests


class EdgeSpeaker:
    """Same interface as pdva.tts.Speaker, backed by an HTTP call to a Jetson."""

    def __init__(self, url, timeout_s=30, health_timeout_s=2):
        self.url = url.rstrip("/")
        self.timeout_s = timeout_s
        self.health_timeout_s = health_timeout_s

    def is_ready(self):
        try:
            r = requests.get(self.url + "/health", timeout=self.health_timeout_s)
            return r.ok
        except requests.RequestException:
            return False

    def synthesize(self, text, out_path):
        """POST text to the Jetson, write the returned wav to out_path.

        Raises requests.RequestException on network failure/timeout,
        and RuntimeError if the Jetson reports a synthesis error, so
        the caller can fall back to the local Speaker if desired.
        """
        r = requests.post(
            self.url + "/synthesize",
            json={"text": text},
            timeout=self.timeout_s,
        )
        if not r.ok:
            detail = (
                r.json().get("error", r.text)
                if r.headers.get("content-type", "").startswith("application/json")
                else r.text
            )
            raise RuntimeError(f"edge synthesize failed ({r.status_code}): {detail}")

        with open(out_path, "wb") as f:
            f.write(r.content)
        return out_path
