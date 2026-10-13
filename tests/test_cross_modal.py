import numpy as np

from deepfake_detection.xai.cross_modal import lip_audio_alignment, mouth_cam_share


def test_mouth_cam_share_is_one_when_all_evidence_sits_on_the_mouth():
    cam = np.zeros((2, 4, 4))
    cam[:, 0, 0] = 1.0
    share, uniform = mouth_cam_share(cam, [(0.0, 0.0, 0.25, 0.25)] * 2)
    assert share == 1.0 and uniform == 1 / 16
    elsewhere, _ = mouth_cam_share(cam, [(0.5, 0.5, 1.0, 1.0)] * 2)
    assert elsewhere == 0.0


def test_lip_audio_alignment_is_high_when_audio_evidence_follows_lip_motion():
    motion = np.zeros(15)
    motion[[4, 5, 10, 11]] = 1.0
    mouth_open = np.concatenate([[0.0], np.cumsum(motion)])
    cam = np.zeros(1500)
    cam[25:37] = 1.0
    cam[62:75] = 1.0
    result = lip_audio_alignment(mouth_open, fps=8.0, audio_step_cam=cam)
    assert result.correlation > 0.99
    assert result.lip_motion.shape == (8,) and result.audio_relevance.shape == (8,)


def test_lip_audio_alignment_is_undefined_for_a_still_mouth():
    result = lip_audio_alignment(np.full(16, 0.3), fps=8.0, audio_step_cam=np.random.rand(1500))
    assert np.isnan(result.correlation)
