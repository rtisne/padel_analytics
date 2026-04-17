"""
Strike posture analysis for padel players.

Computes joint angles from player pose keypoints and evaluates
whether the posture matches good padel form for different shot types.
"""

import math
from dataclasses import dataclass
from typing import Optional


def _angle_between_points(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
) -> float:
    """
    Compute the angle (in degrees) at point *b* formed by segments b→a and b→c.

    Parameters:
        a: first endpoint
        b: vertex
        c: second endpoint

    Returns:
        angle in degrees [0, 180]
    """
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])

    dot = ba[0] * bc[0] + ba[1] * bc[1]
    mag_ba = math.hypot(*ba)
    mag_bc = math.hypot(*bc)

    if mag_ba == 0 or mag_bc == 0:
        return 0.0

    cos_angle = max(-1.0, min(1.0, dot / (mag_ba * mag_bc)))
    return math.degrees(math.acos(cos_angle))


def _vertical_angle(
    top: tuple[float, float],
    bottom: tuple[float, float],
) -> float:
    """
    Compute the angle (in degrees) that the segment *bottom→top* makes with
    the vertical axis.  0° means perfectly upright, 90° means horizontal.
    """
    dx = top[0] - bottom[0]
    dy = top[1] - bottom[1]  # positive = downward in image coords
    # vertical reference is (0, -1) in image coordinates (upward)
    length = math.hypot(dx, dy)
    if length == 0:
        return 0.0
    cos_angle = max(-1.0, min(1.0, -dy / length))
    return math.degrees(math.acos(cos_angle))


# ---------------------------------------------------------------------------
# Ideal angle ranges for good padel posture
# ---------------------------------------------------------------------------

@dataclass
class PostureRange:
    """An acceptable range for a joint angle (degrees)."""
    name: str
    min_deg: float
    max_deg: float
    description: str

    def evaluate(self, angle: float) -> str:
        """Return 'good', 'acceptable', or 'poor'."""
        if self.min_deg <= angle <= self.max_deg:
            return "good"
        # Allow a 15° tolerance band around the ideal range
        if (self.min_deg - 15) <= angle <= (self.max_deg + 15):
            return "acceptable"
        return "poor"


# Ready position (waiting / split-step)
READY_POSITION_RANGES = {
    "left_knee": PostureRange("left_knee", 110, 160, "Slight knee bend for explosiveness"),
    "right_knee": PostureRange("right_knee", 110, 160, "Slight knee bend for explosiveness"),
    "torso_lean": PostureRange("torso_lean", 0, 20, "Torso nearly upright"),
}

# Forehand / backhand strike
STRIKE_RANGES = {
    "left_knee": PostureRange("left_knee", 90, 150, "Knees bent for power transfer"),
    "right_knee": PostureRange("right_knee", 90, 150, "Knees bent for power transfer"),
    "left_elbow": PostureRange("left_elbow", 90, 170, "Arm extension at contact"),
    "right_elbow": PostureRange("right_elbow", 90, 170, "Arm extension at contact"),
    "torso_lean": PostureRange("torso_lean", 0, 30, "Slight forward lean allowed"),
}


# ---------------------------------------------------------------------------
# Per-frame posture metrics
# ---------------------------------------------------------------------------

@dataclass
class PostureMetrics:
    """Joint angles computed from a single set of player keypoints."""

    left_knee_angle: Optional[float] = None
    right_knee_angle: Optional[float] = None
    left_elbow_angle: Optional[float] = None
    right_elbow_angle: Optional[float] = None
    left_shoulder_angle: Optional[float] = None
    right_shoulder_angle: Optional[float] = None
    torso_lean_angle: Optional[float] = None
    shoulder_alignment_angle: Optional[float] = None

    def as_dict(self, prefix: str = "") -> dict[str, Optional[float]]:
        return {
            f"{prefix}left_knee_angle": self.left_knee_angle,
            f"{prefix}right_knee_angle": self.right_knee_angle,
            f"{prefix}left_elbow_angle": self.left_elbow_angle,
            f"{prefix}right_elbow_angle": self.right_elbow_angle,
            f"{prefix}left_shoulder_angle": self.left_shoulder_angle,
            f"{prefix}right_shoulder_angle": self.right_shoulder_angle,
            f"{prefix}torso_lean_angle": self.torso_lean_angle,
            f"{prefix}shoulder_alignment_angle": self.shoulder_alignment_angle,
        }

    def evaluate(self, ranges: dict[str, PostureRange]) -> dict[str, str]:
        """Evaluate each available angle against the given ideal ranges."""
        results = {}
        mapping = {
            "left_knee": self.left_knee_angle,
            "right_knee": self.right_knee_angle,
            "left_elbow": self.left_elbow_angle,
            "right_elbow": self.right_elbow_angle,
            "torso_lean": self.torso_lean_angle,
        }
        for key, angle in mapping.items():
            if angle is not None and key in ranges:
                results[key] = ranges[key].evaluate(angle)
        return results


def compute_posture_metrics(keypoints_by_name: dict) -> PostureMetrics:
    """
    Compute posture metrics from a dictionary of keypoints indexed by name.

    Parameters:
        keypoints_by_name: mapping from keypoint name to an object with an
                           ``xy`` attribute (tuple[float, float]).

    Returns:
        PostureMetrics with all computable angles filled in.
    """

    def _get(name: str) -> Optional[tuple[float, float]]:
        kp = keypoints_by_name.get(name)
        if kp is None:
            return None
        xy = kp.xy
        # Filter out zero-coordinate keypoints (undetected)
        if xy[0] == 0.0 and xy[1] == 0.0:
            return None
        return xy

    metrics = PostureMetrics()

    # Knee angles: hip–knee–foot
    torso = _get("torso")
    left_knee = _get("left_knee")
    left_foot = _get("left_foot")
    right_knee = _get("right_knee")
    right_foot = _get("right_foot")

    if torso and left_knee and left_foot:
        metrics.left_knee_angle = _angle_between_points(torso, left_knee, left_foot)
    if torso and right_knee and right_foot:
        metrics.right_knee_angle = _angle_between_points(torso, right_knee, right_foot)

    # Elbow angles: shoulder–elbow–hand
    left_shoulder = _get("left_shoulder")
    right_shoulder = _get("right_shoulder")
    left_elbow = _get("left_elbow")
    right_elbow = _get("right_elbow")
    left_hand = _get("left_hand")
    right_hand = _get("right_hand")

    if left_shoulder and left_elbow and left_hand:
        metrics.left_elbow_angle = _angle_between_points(
            left_shoulder, left_elbow, left_hand,
        )
    if right_shoulder and right_elbow and right_hand:
        metrics.right_elbow_angle = _angle_between_points(
            right_shoulder, right_elbow, right_hand,
        )

    # Shoulder angles: elbow–shoulder–torso
    if left_elbow and left_shoulder and torso:
        metrics.left_shoulder_angle = _angle_between_points(
            left_elbow, left_shoulder, torso,
        )
    if right_elbow and right_shoulder and torso:
        metrics.right_shoulder_angle = _angle_between_points(
            right_elbow, right_shoulder, torso,
        )

    # Torso lean: angle of head→torso segment with vertical
    head = _get("head")
    if head and torso:
        metrics.torso_lean_angle = _vertical_angle(head, torso)

    # Shoulder alignment: angle of the line between left and right shoulder
    # relative to horizontal (0° = perfectly level).
    # We use abs() because we only care about the magnitude of tilt,
    # not the direction (left-tilt vs right-tilt).
    if left_shoulder and right_shoulder:
        dx = right_shoulder[0] - left_shoulder[0]
        dy = right_shoulder[1] - left_shoulder[1]
        metrics.shoulder_alignment_angle = abs(math.degrees(math.atan2(dy, dx)))

    return metrics


def posture_score(evaluations: dict[str, str]) -> float:
    """
    Compute a 0–100 posture score from evaluation results.

    - 'good'       → 100
    - 'acceptable' → 60
    - 'poor'       → 20
    """
    if not evaluations:
        return 0.0

    score_map = {"good": 100, "acceptable": 60, "poor": 20}
    total = sum(score_map.get(v, 0) for v in evaluations.values())
    return total / len(evaluations)
