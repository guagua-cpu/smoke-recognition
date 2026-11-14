import unittest

import numpy as np

from behavior_rules import classify_pose, defecating_pose, hand_near_mouth, urinating_pose


def _person(points: dict[int, tuple[float, float]], conf: float = 1.0) -> np.ndarray:
    kpts = np.zeros((17, 3), dtype=np.float32)
    for index, (x, y) in points.items():
        kpts[index] = (x, y, conf)
    return kpts


def standing_body() -> dict[int, tuple[float, float]]:
    return {
        0: (200, 40),
        5: (160, 80),
        6: (240, 80),
        7: (150, 140),
        8: (250, 140),
        9: (155, 190),
        10: (245, 190),
        11: (175, 200),
        12: (225, 200),
        13: (170, 280),
        14: (230, 280),
        15: (168, 360),
        16: (232, 360),
    }


class BehaviorRuleTest(unittest.TestCase):
    def test_standing_is_not_an_alarm(self):
        kpts = _person(standing_body())
        flags = classify_pose(kpts, cigarette_hit=True)
        self.assertFalse(any(flags.values()))

    def test_hand_to_mouth_is_smoking_only_with_cigarette(self):
        points = standing_body()
        points[6] = (240, 80)
        points[8] = (260, 40)
        points[10] = (205, 42)
        kpts = _person(points)
        self.assertTrue(hand_near_mouth(kpts))
        self.assertFalse(classify_pose(kpts, cigarette_hit=False)["smoking"])
        self.assertTrue(classify_pose(kpts, cigarette_hit=True)["smoking"])
        self.assertFalse(classify_pose(kpts, cigarette_hit=True)["urinating"])

    def test_one_hand_at_groin_while_standing_is_urinating(self):
        points = standing_body()
        points[8] = (230, 160)
        points[10] = (202, 224)
        kpts = _person(points)
        self.assertTrue(urinating_pose(kpts))
        flags = classify_pose(kpts, cigarette_hit=True)
        self.assertTrue(flags["urinating"])
        self.assertFalse(flags["smoking"])
        self.assertFalse(flags["defecating"])

    def test_deep_squat_is_defecating(self):
        points = {
            0: (200, 30),
            5: (170, 60),
            6: (230, 60),
            7: (150, 100),
            8: (250, 100),
            9: (160, 140),
            10: (240, 140),
            11: (180, 170),
            12: (220, 170),
            13: (140, 185),
            14: (260, 185),
            15: (175, 200),
            16: (225, 200),
        }
        kpts = _person(points)
        self.assertTrue(defecating_pose(kpts))
        flags = classify_pose(kpts, cigarette_hit=True)
        self.assertTrue(flags["defecating"])
        self.assertFalse(flags["urinating"])
        self.assertFalse(flags["smoking"])

    def test_low_confidence_keypoints_do_not_fire(self):
        kpts = _person(standing_body(), conf=0.1)
        kpts[10] = (202, 224, 0.1)
        flags = classify_pose(kpts, cigarette_hit=True)
        self.assertFalse(any(flags.values()))


if __name__ == "__main__":
    unittest.main()
