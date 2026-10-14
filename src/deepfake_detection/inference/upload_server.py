from __future__ import annotations

import base64
import json
import re
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2

from deepfake_detection.inference.multi_subject import VideoResult

ALLOWED_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
MAX_UPLOAD_BYTES = 500 * 1024 * 1024
RED, BLUE = (60, 60, 220), (200, 110, 30)

PAGE = """<!doctype html>
<html lang="id"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Deteksi Deepfake</title>
<style>
body{font-family:Segoe UI,Arial,sans-serif;max-width:960px;margin:2rem auto;padding:0 1rem;color:#111;background:#fff}
h1{font-size:1.4rem}
button,input[type=file]{font:inherit}
button{padding:.5rem 1.2rem;border:1px solid #111;background:#fff;cursor:pointer}
button:disabled{opacity:.5;cursor:default}
table{border-collapse:collapse;margin:1rem 0;width:100%}
th,td{border:1px solid #bbb;padding:.4rem .6rem;text-align:left}
.palsu{color:#c0392b;font-weight:600}.asli{color:#1f6fb2;font-weight:600}
img{max-width:100%;border:1px solid #bbb}
#status{margin:.8rem 0;min-height:1.4rem}
</style></head><body>
<h1>Deteksi deepfake audio-visual</h1>
<p>Unggah satu video (mp4, mov, mkv, avi, webm). Setiap wajah mendapat putusan sendiri; video dinyatakan palsu bila ada satu wajah yang palsu.</p>
<input id="file" type="file" accept="video/*"> <button id="go" disabled>Periksa</button>
<div id="status"></div>
<div id="result"></div>
<script>
const file = document.getElementById('file'), go = document.getElementById('go'), status = document.getElementById('status'), result = document.getElementById('result');
file.onchange = () => { go.disabled = !file.files.length; };
go.onclick = async () => {
  const f = file.files[0]; if (!f) return;
  go.disabled = true; result.innerHTML = '';
  const t0 = Date.now(), timer = setInterval(() => { status.textContent = 'Memproses ' + f.name + ' ... ' + Math.round((Date.now() - t0) / 1000) + ' detik'; }, 500);
  try {
    const r = await fetch('/score?name=' + encodeURIComponent(f.name), {method: 'POST', body: f});
    const d = await r.json();
    if (d.error) { status.textContent = 'Gagal: ' + d.error; return; }
    status.textContent = 'Selesai dalam ' + d.seconds + ' detik';
    let html = '';
    if (!d.faces.length) html += '<p>Tidak ada wajah yang terlihat penuh selama satu segmen 2 detik.</p>';
    else {
      html += '<p>Putusan video: <span class="' + d.video_verdict + '">' + d.video_verdict.toUpperCase() + '</span> (P(palsu) ' + d.video_p_fake.toFixed(4) + ', ambang ' + d.threshold + ')</p>';
      html += '<table><tr><th>Orang</th><th>Pembicara</th><th>Segmen</th><th>P(palsu)</th><th>&alpha;</th><th>Putusan</th></tr>';
      for (const p of d.faces) html += '<tr><td>' + p.label + '</td><td>' + (p.speaker || '-') + '</td><td>' + p.segments + '</td><td>' + p.p_fake.toFixed(4) + '</td><td>' + (p.alpha === null ? '-' : p.alpha.toFixed(3)) + '</td><td class="' + p.verdict + '">' + p.verdict + '</td></tr>';
      html += '</table>';
    }
    if (d.preview) html += '<img src="' + d.preview + '" alt="Wajah yang terdeteksi">';
    result.innerHTML = html;
  } catch (e) { status.textContent = 'Gagal: ' + e; }
  finally { clearInterval(timer); go.disabled = false; }
};
</script></body></html>
"""


def person_labels(result: VideoResult) -> dict[int, str]:
    ordered = sorted(result.subjects, key=lambda s: s.center_x)
    return {s.track_id: f"Orang {chr(ord('A') + i)}" for i, s in enumerate(ordered)}


def preview_image(path: Path, result: VideoResult, labels: dict[int, str]) -> str | None:
    """The middle frame of the video with each face box and its P(fake), as a data URL."""
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // 2, 0))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    scale = min(1.0, 960 / frame.shape[1])
    if scale < 1.0:
        frame = cv2.resize(frame, None, fx=scale, fy=scale)
    for subject in result.subjects:
        if subject.box is None:
            continue
        x0, y0, x1, y1 = (int(v * scale) for v in subject.box)
        color = RED if subject.flagged else BLUE
        cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
        cv2.putText(frame, f"{labels[subject.track_id]} {subject.p_fake:.2f}", (x0, max(y0 - 8, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode("ascii") if ok else None


def build_payload(result: VideoResult, name: str, seconds: float, preview: str | None = None) -> dict:
    labels = person_labels(result)
    faces = [
        {"label": labels[s.track_id], "speaker": s.speaker, "segments": len(s.segments), "p_fake": round(s.p_fake, 4),
         "alpha": None if s.alpha is None else round(s.alpha, 3), "verdict": "palsu" if s.flagged else "asli"}
        for s in sorted(result.subjects, key=lambda s: s.center_x)
    ]
    verdict = "palsu" if result.subjects and result.video_p_fake >= result.threshold else "asli"
    return {"name": name, "seconds": round(seconds, 1), "faces": faces, "video_p_fake": result.video_p_fake if result.subjects else None,
            "threshold": result.threshold, "video_verdict": verdict, "duration": result.duration,
            "turns": [[t.start, t.end, t.speaker] for t in result.turns], "preview": preview}


def make_handler(pipeline, upload_dir: Path, out_dir: Path, keep_uploads: bool):
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            print(f"{time.strftime('%T')} {self.address_string()} {fmt % args}", flush=True)

        def send_body(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, code: int, payload: dict) -> None:
            self.send_body(code, json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8")

        def do_GET(self):
            if urlparse(self.path).path == "/":
                self.send_body(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
            else:
                self.send_json(404, {"error": "not found"})

        def do_POST(self):
            parsed = urlparse(self.path)
            if parsed.path != "/score":
                return self.send_json(404, {"error": "not found"})
            name = re.sub(r"[^A-Za-z0-9._-]", "_", Path(parse_qs(parsed.query).get("name", ["video.mp4"])[0]).name)
            if Path(name).suffix.lower() not in ALLOWED_SUFFIXES:
                return self.send_json(400, {"error": f"file type not accepted, use one of {sorted(ALLOWED_SUFFIXES)}"})
            length = int(self.headers.get("Content-Length", 0))
            if not 0 < length <= MAX_UPLOAD_BYTES:
                return self.send_json(413 if length > 0 else 400, {"error": f"size must be between 1 byte and {MAX_UPLOAD_BYTES // 2**20} MB"})
            target = upload_dir / f"{uuid.uuid4().hex[:8]}_{name}"
            with open(target, "wb") as f:
                remaining = length
                while remaining > 0:
                    chunk = self.rfile.read(min(1 << 20, remaining))
                    if not chunk:
                        break
                    f.write(chunk)
                    remaining -= len(chunk)
            try:
                with lock:
                    started = time.time()
                    result = pipeline.run(target)
                    seconds = time.time() - started
                payload = build_payload(result, name, seconds)
                (out_dir / f"{target.stem}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
                payload["preview"] = preview_image(target, result, person_labels(result)) if result.subjects else None
                self.send_json(200, payload)
            except Exception as error:
                self.send_json(500, {"error": f"{type(error).__name__}: {error}"})
            finally:
                if not keep_uploads:
                    target.unlink(missing_ok=True)

    return Handler


def serve(pipeline, host: str, port: int, upload_dir: Path, out_dir: Path, keep_uploads: bool = False) -> None:
    upload_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((host, port), make_handler(pipeline, upload_dir, out_dir, keep_uploads))
    print(f"Upload demo at http://{host}:{port}/  (Ctrl-C stops; uploads {'kept in' if keep_uploads else 'deleted after scoring from'} {upload_dir}, results in {out_dir})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("stopped")
    finally:
        server.server_close()
