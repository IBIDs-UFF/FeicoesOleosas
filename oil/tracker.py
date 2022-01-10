from .detector import BoundingBox, DropDetector
from .subtractor import BackgroundSubtractor
from typing import Iterator, List, TypedDict, Optional, Tuple
import itertools
import numpy as np


BACKGROUND_ID = 0


class Drop(TypedDict):
    id: int
    active: bool
    bbox: BoundingBox
    first_seen: int
    last_seen: int


class DropTracker():

    def __init__(self, subtractor: BackgroundSubtractor, patience: int) -> None:
        # Create a drop detector.
        self._detector = DropDetector(subtractor)
        # Initialize other attributes.
        self._active_drops: List[Drop] = list()
        self._former_drops: List[Drop] = list()
        self._lost_drops_footprint: Optional[np.ndarray] = None
        self._next_drop_id = 1
        self._patience = int(patience)

    def _do_overlap(self, bbox1: BoundingBox, bbox2: BoundingBox) -> bool:
        l1_x, l1_y, r1_x, r1_y = bbox1.x, bbox1.y + bbox1.h, bbox1.x + bbox1.w, bbox1.y
        l2_x, l2_y, r2_x, r2_y = bbox2.x, bbox2.y + bbox2.h, bbox2.x + bbox2.w, bbox2.y
        if l1_x >= r2_x or l2_x >= r1_x:
            return False
        if r1_y >= l2_y or r2_y >= l1_y:
            return False
        return True

    def update(self, frame_rgb: np.ndarray, frame_gray: np.ndarray, frame_ind: int) -> None:
        height, width = frame_gray.shape
        drops = self._detector.detect(frame_rgb, frame_gray)
        for drop_bbox, may_be_new in drops:
            match_not_found = True
            for active_drop in self._active_drops:
                if self._do_overlap(drop_bbox, active_drop['bbox']):
                    active_drop['bbox'] = drop_bbox
                    active_drop['last_seen'] = frame_ind
                    match_not_found = False
                    break
            if match_not_found and may_be_new:
                self._active_drops.append(Drop(
                    id=self._next_drop_id,
                    active=True,
                    bbox=drop_bbox,
                    first_seen=frame_ind,
                    last_seen=frame_ind,
                ))
                self._next_drop_id += 1
        lost_drops: List[Tuple[int, BoundingBox]] = list()
        for drop_ind in range(len(self._active_drops) - 1, -1, -1):
            drop = self._active_drops[drop_ind]
            if frame_ind - drop['last_seen'] >= self._patience:
                drop['active'] = False
                self._former_drops.append(drop)
                del self._active_drops[drop_ind]
                lost_drops.append((drop['id'], drop['bbox']))
        if len(lost_drops) > 0:
            self._lost_drops_footprint = np.full((height, width), BACKGROUND_ID, dtype=np.int32)
            for drop_id, drop_bbox in lost_drops:
                x1 = max(int(drop_bbox.x) - 5, 0)
                x2 = min(int(drop_bbox.x + drop_bbox.w) + 5, width - 1)
                y1 = max(int(drop_bbox.y) - 5, 0)
                y2 = min(int(drop_bbox.y + drop_bbox.h) + 5, height - 1)
                self._lost_drops_footprint[y1:y2, x1:x2] = drop_id
        else:
            self._lost_drops_footprint = None

    @property
    def drops(self) -> Iterator[Drop]:
        return itertools.chain(self._former_drops, self._active_drops)

    @property
    def background_msk(self) -> np.ndarray:
        return self._detector.background_msk

    @property
    def lost_drops_footprint(self) -> Optional[np.ndarray]:
        return self._lost_drops_footprint

    @property
    def patience(self) -> int:
        return self._patience
