from numpy import argmin
from .drop_detector import BoundingBox, DropDetector
from torch import Tensor
from typing import Iterator, List, TypedDict, Optional, Tuple
import itertools
import torch


BACKGROUND_ID = 0


class Drop(TypedDict):
    id: Optional[int]
    active: bool
    bbox: BoundingBox
    confidence: float
    first_seen: int
    last_seen: int


class DropTracker():

    @torch.no_grad()
    def __init__(self, *, confidence_threshold: float, device: torch.device, frame_size: int, lifetime_threshold: int, patience: int) -> None:
        # Create a drop detector.
        self._detector = DropDetector(confidence_threshold=confidence_threshold, device=device, frame_size=frame_size, iou_threshold=0.45)
        # Initialize other attributes.
        self._active_drops: List[Drop] = list()
        self._former_drops: List[Drop] = list()
        self._lifetime_threshold = lifetime_threshold
        self._lost_drops_footprint: Optional[Tensor] = None
        self._next_drop_id = BACKGROUND_ID + 1
        self._patience = int(patience)

    @torch.no_grad()
    def _do_overlap(self, bbox1: BoundingBox, bbox2: BoundingBox) -> bool:
        l1_x, l1_y, r1_x, r1_y = bbox1.x, bbox1.y + bbox1.h, bbox1.x + bbox1.w, bbox1.y
        l2_x, l2_y, r2_x, r2_y = bbox2.x, bbox2.y + bbox2.h, bbox2.x + bbox2.w, bbox2.y
        if l1_x >= r2_x or l2_x >= r1_x:
            return False
        if r1_y >= l2_y or r2_y >= l1_y:
            return False
        return True

    @torch.no_grad()
    def update(self, frame_rgb: Tensor, frame_ind: int) -> None:
        _, height, width = frame_rgb.shape
        drops = self._detector.detect(frame_rgb)
        for drop_bbox, drop_confidence in drops:
            drop_area = drop_bbox.w * drop_bbox.h
            candidates: List[Tuple[Drop, float]] = []
            for other_drop in self._active_drops:
                other_bbox = other_drop['bbox']
                other_area = other_bbox.w * other_bbox.h
                area_ratio = other_area / drop_area if other_area > drop_area else drop_area / other_area
                if area_ratio < 2.0 and self._do_overlap(drop_bbox, other_bbox):
                    candidates.append((other_drop, other_area))
            if len(candidates) > 0:
                matching_drop, _ = min(candidates, key=lambda candidate: abs(drop_area - candidate[1]))
                matching_drop['bbox'] = drop_bbox
                matching_drop['confidence'] = drop_confidence
                matching_drop['last_seen'] = frame_ind
            else:
                self._active_drops.append(Drop(id=None, active=True, bbox=drop_bbox, confidence=drop_confidence, first_seen=frame_ind, last_seen=frame_ind))
        lost_drops: List[Tuple[int, BoundingBox]] = list()
        for drop_ind in range(len(self._active_drops) - 1, -1, -1):
            drop = self._active_drops[drop_ind]
            if frame_ind - drop['last_seen'] >= self._patience:
                del self._active_drops[drop_ind]
                if drop['last_seen'] - drop['first_seen'] >= self._lifetime_threshold:
                    drop['id'] = self._next_drop_id
                    drop['active'] = False
                    self._former_drops.append(drop)
                    lost_drops.append((drop['id'], drop['bbox']))
                    self._next_drop_id += 1
        if len(lost_drops) > 0:
            self._lost_drops_footprint = torch.full((height, width), BACKGROUND_ID, dtype=torch.int32, device=frame_rgb.device)
            for drop_id, drop_bbox in lost_drops:
                x1 = max(int(drop_bbox.x - drop_bbox.w), 0)
                x2 = min(int(drop_bbox.x + 2 * drop_bbox.w), width - 1)
                y1 = max(int(drop_bbox.y - drop_bbox.w), 0)
                y2 = min(int(drop_bbox.y + 2 * drop_bbox.h), height - 1)
                self._lost_drops_footprint[y1:y2, x1:x2] = drop_id
        else:
            self._lost_drops_footprint = None

    @property
    def drops(self) -> Iterator[Drop]:
        return itertools.chain(self._former_drops, self._active_drops)

    @property
    def lost_drops_footprint(self) -> Optional[Tensor]:
        return self._lost_drops_footprint

    @property
    def patience(self) -> int:
        return self._patience
