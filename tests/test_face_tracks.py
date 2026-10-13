from deepfake_detection.data.face_crop import FaceTrack, box_iou, link_boxes


def box(x, y, size=100):
    return (x, y, x + size, y + size)


def test_iou_identical_and_disjoint():
    assert box_iou(box(0, 0), box(0, 0)) == 1.0
    assert box_iou(box(0, 0), box(500, 500)) == 0.0


def test_two_faces_stay_two_tracks_while_moving():
    detections = {
        0: [box(0, 0), box(400, 0)],
        10: [box(10, 0), box(390, 0)],
        20: [box(20, 0), box(380, 0)],
    }
    tracks = link_boxes(detections)
    assert len(tracks) == 2
    assert sorted(len(t.detections) for t in tracks) == [3, 3]
    left = min(tracks, key=lambda t: t.detections[0][0])
    assert left.detections[20] == box(20, 0)


def test_new_face_starts_a_new_track_and_a_gap_keeps_the_old_one():
    detections = {0: [box(0, 0)], 10: [], 20: [box(5, 0), box(400, 0)]}
    tracks = link_boxes(detections)
    assert len(tracks) == 2
    first = next(t for t in tracks if 0 in t.detections)
    assert sorted(first.detections) == [0, 20]
    second = next(t for t in tracks if 0 not in t.detections)
    assert sorted(second.detections) == [20]


def test_track_interpolates_between_detections():
    track = FaceTrack(0, {0: box(0, 0), 10: box(100, 0)})
    boxes = track.boxes_for_indices([0, 5, 10])
    assert boxes[5][0] == 50.0
    assert boxes[10] == box(100, 0)
