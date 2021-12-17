from .detector import DropDetector
from typing import Any, Dict, Iterator, List, Optional, Tuple
import itertools
import numpy as np


class DropTracker():

    def __init__(self, background_subtractor: Any, patience: int) -> None:
        # Create a drop detector.
        self._detector = DropDetector(background_subtractor)
        # Initialize other attributes.
        self._active_drops: List[Dict[str, Any]] = list()
        self._former_drops: List[Dict[str, Any]] = list()
        self._lost_drops_footprint: Optional[np.ndarray] = None
        self._next_drop_id = 1
        self._patience = int(patience)

    def _do_overlap(self, rect1: Tuple[float, float, float, float], rect2: Tuple[float, float, float, float]) -> bool:
        l1_x, l1_y, r1_x, r1_y = rect1[0], rect1[1] + rect1[3], rect1[0] + rect1[2], rect1[1]
        l2_x, l2_y, r2_x, r2_y = rect2[0], rect2[1] + rect2[3], rect2[0] + rect2[2], rect2[1]
        if l1_x >= r2_x or l2_x >= r1_x:
            return False
        if r1_y >= l2_y or r2_y >= l1_y:
            return False
        return True

    def update(self, frame_rgb: np.ndarray, frame_gray: np.ndarray, frame_ind: int) -> None:
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
                self._active_drops.append({
                    'id': self._next_drop_id,
                    'active': True,
                    'bbox': drop_bbox,
                    'first_seen': frame_ind,
                    'last_seen': frame_ind,
                })
                self._next_drop_id += 1
        lost_drops = list()
        for drop_ind in range(len(self._active_drops) - 1, -1, -1):
            drop = self._active_drops[drop_ind]
            if frame_ind - drop['last_seen'] > self._patience:
                drop['active'] = False
                self._former_drops.append(drop)
                del self._active_drops[drop_ind]
                lost_drops.append((active_drop['id'], active_drop['bbox']))
        if len(lost_drops) > 0:
            self._lost_drops_footprint = np.ndarray(frame_gray.shape, dtype=np.int32)
            for drop_id, drop_bbox in lost_drops:
                self._lost_drops_footprint[drop_bbox[1]:drop_bbox[1]+drop_bbox[3], drop_bbox[0]:drop_bbox[0]+drop_bbox[2]] = drop_id
        else:
            self._lost_drops_footprint = None

    @property
    def drops(self) -> Iterator[Dict[str, Any]]:
        return itertools.chain(self._former_drops, self._active_drops)

    @property
    def drop_msk(self) -> np.ndarray:
        return self._detector.drop_msk

    @property
    def foreground_msk(self) -> np.ndarray:
        return self._detector.foreground_msk

    @property
    def lost_drops_footprint(self) -> Optional[np.ndarray]:
        return self._lost_drops_footprint

    @property
    def patience(self) -> int:
        return self._patience

    @property
    def shadow_msk(self) -> np.ndarray:
        return self._detector.shadow_msk
