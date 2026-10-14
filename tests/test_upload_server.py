import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import cv2
import numpy as np
import pytest

from deepfake_detection.inference import SubjectResult, aggregate_subjects
from deepfake_detection.inference.upload_server import make_handler


class FakePipeline:
    def __init__(self):
        self.seen = []

    def run(self, path):
        self.seen.append(path)
        subject = SubjectResult(track_id=0, center_x=0.5, segments=[0, 1], p_fake=0.9, alpha=0.2, speaker=None, flagged=True, box=(10.0, 10.0, 40.0, 40.0))
        return aggregate_subjects([subject], threshold=0.5, duration=4.0)


@pytest.fixture
def server(tmp_path):
    pipeline = FakePipeline()
    upload_dir, out_dir = tmp_path / "uploads", tmp_path / "out"
    upload_dir.mkdir()
    out_dir.mkdir()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(pipeline, upload_dir, out_dir, keep_uploads=False))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", pipeline, upload_dir, out_dir
    httpd.shutdown()
    httpd.server_close()


def tiny_video(path):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 64))
    for i in range(10):
        writer.write(np.full((64, 64, 3), i * 20, dtype=np.uint8))
    writer.release()
    return path.read_bytes()


def post(url, data):
    request = urllib.request.Request(url, data=data, method="POST", headers={"Content-Type": "application/octet-stream"})
    return json.loads(urllib.request.urlopen(request, timeout=30).read())


def test_page_is_served(server):
    base, *_ = server
    assert b"Deteksi deepfake" in urllib.request.urlopen(base + "/", timeout=10).read()


def test_upload_is_scored_saved_and_deleted(server, tmp_path):
    base, pipeline, upload_dir, out_dir = server
    payload = post(base + "/score?name=clip%20one.mp4", tiny_video(tmp_path / "v.mp4"))
    assert payload["video_verdict"] == "palsu"
    assert payload["faces"] == [{"label": "Orang A", "speaker": None, "segments": 2, "p_fake": 0.9, "alpha": 0.2, "verdict": "palsu"}]
    assert payload["preview"].startswith("data:image/jpeg;base64,")
    assert len(pipeline.seen) == 1 and pipeline.seen[0].name.endswith("clip_one.mp4")
    assert list(upload_dir.iterdir()) == []
    saved = list(out_dir.glob("*.json"))
    assert len(saved) == 1
    assert json.loads(saved[0].read_text(encoding="utf-8"))["preview"] is None


def test_other_file_types_are_refused(server):
    base, pipeline, *_ = server
    with pytest.raises(urllib.error.HTTPError) as error:
        post(base + "/score?name=notes.txt", b"hello")
    assert error.value.code == 400 and pipeline.seen == []
