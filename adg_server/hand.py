"""Two-finger grasps to Inspire-hand poses: InspireHandRGraspGroup.graspgroupTR_2_TR in one pass.

The original rebuilds the list of every grasp's type name for each grasp, three times: quadratic, ~8 s for 2500
grasps.
"""

import json
import os

import numpy as np


class HandMap:
    """width_12Dangle_6Dangle.json: per hand type and width (cm, 1 mm steps) the hand pose relative to the two-finger
    pose and the preshape."""

    def __init__(self, adg, mesh_dir: str):
        with open(os.path.join(mesh_dir, "width_12Dangle_6Dangle.json"), encoding="utf-8") as f:
            self.mapping = json.load(f)
        self.types = adg.grasp_types
        self.min_width, self.max_width = adg.inspire.MIN_GRASP_WIDTH, adg.inspire.MAX_GRASP_WIDTH

    def names(self, grasp_types: np.ndarray) -> list[str]:
        return [self.types[str(int(t))]["name"] for t in grasp_types]

    def apply(self, hand, two) -> None:
        """Set `hand`'s widths, poses and preshapes from the two-finger grasps `two` (row for row)."""
        widths, translations, rotations, angles = [], [], [], []
        for name, grasp_type, width, t, r in zip(
            self.names(hand.grasp_types), hand.grasp_types, two.widths, two.translations, two.rotation_matrices
        ):
            lo, hi = self.types[str(int(grasp_type))]["width"]
            width = min(min(hi, self.max_width), max(max(lo, self.min_width), width))
            entry = self.mapping[name][str(np.round(width * 100, 1))]
            two_from_hand = np.eye(4)
            two_from_hand[:3, :3], two_from_hand[:3, 3] = entry["rotation"], entry["translation"]
            pose = np.eye(4)
            pose[:3, :3], pose[:3, 3] = r, t
            pose = pose @ np.linalg.inv(two_from_hand)
            widths.append(width)
            translations.append(pose[:3, 3])
            rotations.append(pose[:3, :3])
            angles.append(entry["6d"])
        hand.widths = np.array(widths)
        hand.translations = np.array(translations)
        hand.rotation_matrices = np.array(rotations)
        hand.angles = np.array(angles)
