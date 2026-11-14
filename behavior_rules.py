"""Pose rules for smoking, public urination, and public defecation.

Geometry follows the smoking project this repo started from: elbow angle and
hand-to-nose distance, scaled by torso length. Urination and defecation use
the same COCO-17 keypoints, with a standing groin-hand pose and a deep squat.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

NOSE = 0
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16

ACTION_LABELS = {
    "smoking": "抽烟",
    "urinating": "随地小便",
    "defecating": "随地大便",
}

# How many consecutive hit-frames before an alarm. Defecation is slower on
# purpose so a person tying a shoe does not trip it as easily.
DEFAULT_THRESHOLDS = {
    "smoking": 8,
    "urinating": 10,
    "defecating": 12,
}
POSE_ONLY_SMOKING_THRESHOLD = 18


def _ok(kpts: np.ndarray, idx: int, conf: float = 0.35) -> bool:
    return float(kpts[idx, 2]) >= conf


def _pt(kpts: np.ndarray, idx: int) -> np.ndarray:
    return np.array([kpts[idx, 0], kpts[idx, 1]], dtype=np.float32)


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> Optional[float]:
    v1 = a - b
    v2 = c - b
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-3 or n2 < 1e-3:
        return None
    cos = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return math.degrees(math.acos(cos))


def _mid(points: list[np.ndarray]) -> np.ndarray:
    return np.mean(np.stack(points, axis=0), axis=0)


def torso_length(kpts: np.ndarray) -> Optional[float]:
    shoulders = [_pt(kpts, i) for i in (L_SHOULDER, R_SHOULDER) if _ok(kpts, i)]
    hips = [_pt(kpts, i) for i in (L_HIP, R_HIP) if _ok(kpts, i)]
    if not shoulders or not hips:
        return None
    length = float(np.linalg.norm(_mid(shoulders) - _mid(hips)))
    if length < 8:
        return None
    return length


def hand_near_mouth(kpts: np.ndarray) -> bool:
    """Bent elbow with the wrist close to the nose, relative to torso length."""
    scale = torso_length(kpts)
    if scale is None or not _ok(kpts, NOSE):
        return False
    nose = _pt(kpts, NOSE)
    for shoulder, elbow, wrist in (
        (L_SHOULDER, L_ELBOW, L_WRIST),
        (R_SHOULDER, R_ELBOW, R_WRIST),
    ):
        if not (_ok(kpts, shoulder) and _ok(kpts, elbow) and _ok(kpts, wrist)):
            continue
        dist_ratio = float(np.linalg.norm(nose - _pt(kpts, wrist))) / scale
        ang = _angle(_pt(kpts, shoulder), _pt(kpts, elbow), _pt(kpts, wrist))
        if dist_ratio < 0.85 and ang is not None and ang < 80:
            return True
    return False


def _knee_angle(kpts: np.ndarray, hip: int, knee: int, ankle: int) -> Optional[float]:
    if not (_ok(kpts, hip) and _ok(kpts, knee) and _ok(kpts, ankle)):
        return None
    return _angle(_pt(kpts, hip), _pt(kpts, knee), _pt(kpts, ankle))


def _leg_metrics(kpts: np.ndarray, scale: float) -> tuple[list[float], list[float]]:
    angles: list[float] = []
    shin_ratios: list[float] = []
    for hip, knee, ankle in ((L_HIP, L_KNEE, L_ANKLE), (R_HIP, R_KNEE, R_ANKLE)):
        ang = _knee_angle(kpts, hip, knee, ankle)
        if ang is None:
            continue
        angles.append(ang)
        shin_ratios.append(float(_pt(kpts, ankle)[1] - _pt(kpts, knee)[1]) / scale)
    return angles, shin_ratios


def urinating_pose(kpts: np.ndarray) -> bool:
    """Standing, exactly one wrist held at the groin, legs not folded."""
    scale = torso_length(kpts)
    if scale is None:
        return False
    angles, shin_ratios = _leg_metrics(kpts, scale)
    if not angles or min(angles) < 145:
        return False
    if shin_ratios and min(shin_ratios) < 0.45:
        return False
    hips = [_pt(kpts, i) for i in (L_HIP, R_HIP) if _ok(kpts, i)]
    knees = [_pt(kpts, i) for i in (L_KNEE, R_KNEE) if _ok(kpts, i)]
    if not hips or not knees:
        return False
    hip = _mid(hips)
    knee = _mid(knees)
    groin = hip + 0.28 * (knee - hip)
    near = 0
    for wrist in (L_WRIST, R_WRIST):
        if not _ok(kpts, wrist):
            continue
        wrist_pt = _pt(kpts, wrist)
        if float(wrist_pt[1]) < float(hip[1]) - 0.15 * scale:
            continue
        if float(np.linalg.norm(wrist_pt - groin)) < 0.38 * scale:
            near += 1
    return near == 1


def defecating_pose(kpts: np.ndarray) -> bool:
    """Deep squat: knees folded, shins nearly horizontal, hips dropped."""
    scale = torso_length(kpts)
    if scale is None:
        return False
    angles, shin_ratios = _leg_metrics(kpts, scale)
    if not angles or not shin_ratios:
        return False
    shoulders = [_pt(kpts, i) for i in (L_SHOULDER, R_SHOULDER) if _ok(kpts, i)]
    hips = [_pt(kpts, i) for i in (L_HIP, R_HIP) if _ok(kpts, i)]
    ankles = [_pt(kpts, i) for i in (L_ANKLE, R_ANKLE) if _ok(kpts, i)]
    if not shoulders or not hips or not ankles:
        return False
    shoulder_y = float(_mid(shoulders)[1])
    hip_y = float(_mid(hips)[1])
    ankle_y = float(_mid(ankles)[1])
    body_span = ankle_y - shoulder_y
    if body_span < scale * 0.35:
        return False
    hip_drop = (hip_y - shoulder_y) / body_span
    return min(angles) < 110 and min(shin_ratios) < 0.55 and hip_drop > 0.62


def classify_pose(kpts: np.ndarray, cigarette_hit: bool) -> dict[str, bool]:
    """Return which watched actions are indicated on this single frame.

    ``cigarette_hit`` is the cigarette detector result. When that model is
    unavailable the caller may pass the hand-to-mouth pose itself, which makes
    smoking a pose-only decision and should use a higher alarm threshold.
    """
    kpts = np.asarray(kpts, dtype=np.float32)
    empty = {name: False for name in ACTION_LABELS}
    if kpts.shape != (17, 3):
        return empty
    squatting = defecating_pose(kpts)
    peeing = urinating_pose(kpts) and not squatting
    smoking = hand_near_mouth(kpts) and bool(cigarette_hit) and not squatting and not peeing
    return {
        "smoking": smoking,
        "urinating": peeing,
        "defecating": squatting,
    }
