import cv2
import numpy as np

from deepfake_detection.inference.multi_subject import frame_reader


class FakeCapture:
    """A decoder that returns the frame index in its pixels and counts seeks."""

    def __init__(self, count: int):
        self.count, self.pos, self.seeks, self.grabbed = count, 0, 0, 0

    def get(self, prop):
        return {cv2.CAP_PROP_FRAME_WIDTH: 1000, cv2.CAP_PROP_FRAME_HEIGHT: 1000}.get(prop, 0)

    def set(self, prop, value):
        self.pos, self.seeks = int(value), self.seeks + 1

    def grab(self):
        self.pos, self.grabbed = self.pos + 1, self.grabbed + 1
        return True

    def read(self):
        if self.pos >= self.count:
            return False, None
        frame = np.full((2, 2, 3), self.pos % 256, dtype=np.uint8)
        self.pos += 1
        return True, frame


def test_ascending_indices_are_read_forward_without_seeking():
    cap = FakeCapture(300)  # 300 x 1000 x 1000 x 3 bytes is above the in-memory limit
    read = frame_reader(cap, 300)
    assert [int(read(i)[0, 0, 0]) for i in (0, 3, 6, 40, 41, 200)] == [0, 3, 6, 40, 41, 200]
    assert cap.seeks == 1 and cap.grabbed < 250


def test_going_back_seeks_once_and_a_repeated_index_is_cached():
    cap = FakeCapture(300)
    read = frame_reader(cap, 300)
    read(100)
    seeks = cap.seeks
    assert int(read(10)[0, 0, 0]) == 10 and cap.seeks == seeks + 1
    read(10)
    assert cap.seeks == seeks + 1


def test_a_frame_past_the_end_returns_the_last_decoded_frame():
    cap = FakeCapture(300)
    read = frame_reader(cap, 300)
    last = read(299)
    assert np.array_equal(read(310), last)
