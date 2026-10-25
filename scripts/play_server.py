"""Tiny local helper for the PulseFlow music map.

Run it in the background:   python play_server.py
Then the "Play in default app" button on the map opens tracks in whatever
program Windows uses for that file type. It only listens on this computer
(127.0.0.1) and only opens audio files inside MUSIC_ROOT (or folders inside it).
"""
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

MUSIC_ROOT = r"M:\Music"
PORT = 8765
AUDIO_EXT = {".m4a", ".m4p", ".mp3", ".flac", ".wav", ".aac", ".ogg", ".opus", ".wma", ".aif", ".aiff", ".alac"}


def inside_root(p):
    try:
        root = os.path.normcase(os.path.abspath(MUSIC_ROOT))
        target = os.path.normcase(os.path.abspath(p))
        return os.path.commonpath([root, target]) == root
    except ValueError:
        return False


class Handler(BaseHTTPRequestHandler):
    def _reply(self, code):
        self.send_response(code)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        url = urlparse(self.path)
        raw = parse_qs(url.query).get("p", [""])[0]
        if url.path not in ("/play", "/folder") or not raw or not inside_root(raw):
            return self._reply(403)
        p = Path(raw)
        try:
            if url.path == "/play" and p.is_file() and p.suffix.lower() in AUDIO_EXT:
                os.startfile(str(p))
                print(f"Playing: {p}", flush=True)
                return self._reply(204)
            if url.path == "/folder" and p.is_dir():
                os.startfile(str(p))
                print(f"Opening folder: {p}", flush=True)
                return self._reply(204)
        except OSError as exc:
            print(f"Failed: {exc}", flush=True)
            return self._reply(500)
        return self._reply(404)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    print(f"PulseFlow player helper listening on http://127.0.0.1:{PORT}  (Ctrl+C to stop)", flush=True)
    HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
