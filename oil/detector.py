from kht import kht
from typing import Any, List, NamedTuple, Tuple
import cv2
import math
import numpy as np


class BoundingBox(NamedTuple):
    x: float
    y: float
    w: float
    h: float


class Circle(NamedTuple):
    center: np.ndarray
    radius: float


class DropDetector():

    _MORPH_OPEN_KERNEL = np.ones((5, 5), np.uint8)

    def __init__(self, background_subtractor: Any) -> None:
        # Create the blob detector.
        params = cv2.SimpleBlobDetector_Params()
        params.filterByColor = True
        params.blobColor = 255
        params.filterByArea = True
        params.minArea = 25
        params.filterByCircularity = True
        params.minCircularity = 0.5
        params.filterByConvexity = False
        params.filterByInertia = False
        self._blob_detector = cv2.SimpleBlobDetector_create(params)
        # Initialize other attributes.
        self._background_subtractor = background_subtractor
        self._circles: List[Circle] = list()
        self._drop_msk: np.ndarray = None
        self._foreground_msk: np.ndarray = None
        self._shadow_msk: np.ndarray = None

    def _compute_line_coefficients(self, line: Tuple[float, float], img_size: Tuple[int, int]) -> np.ndarray:
        width, height = img_size
        rho, theta = line
        theta = math.radians(theta)
        cos_theta, sin_theta = math.cos(theta), math.sin(theta)
        if sin_theta != 0:
            x = (-width / 2, width / 2 - 1)
            y = ((rho - x[0] * cos_theta) / sin_theta, (rho - x[1] * cos_theta) / sin_theta)
        else:
            x = (rho, rho)
            y = (-height / 2, height / 2 - 1)
        x = (x[0] + width / 2, x[1] + width / 2)
        y = (y[0] + height / 2, y[1] + height / 2)
        return np.array((y[0] - y[1], x[1] - x[0], x[0] * y[1] - x[1] * y[0]), dtype=np.float32)

    def _may_be_new(self, pt: np.ndarray) -> bool:
        for center, radius in self._circles:
            if np.linalg.norm(pt - center, ord=2) <= radius:
                return True
        return False

    def detect(self, frame_rgb: np.ndarray, frame_gray: np.ndarray) -> List[Tuple[BoundingBox, bool]]:
        # Compute birth regions.
        height, width = frame_gray.shape
        radius = 0.10 * min(width, height)
        self._circles.clear()
        lines = kht(cv2.Canny(frame_gray, 80, 200))
        for line_ind1 in range(3):
            line1 = self._compute_line_coefficients(lines[line_ind1], (width, height))
            for line_ind2 in range(line_ind1 + 1, 3):
                line2 = self._compute_line_coefficients(lines[line_ind2], (width, height))
                pt = np.cross(line1, line2)
                if pt[-1] != 0:
                    pt = pt[:-1] / pt[-1]
                    if 0 <= pt[0] < width and 0 <= pt[1] < height:
                        self._circles.append((pt, radius))
        # Compute foreground, shadow, and drop masks.
        temp_msk = self._background_subtractor.apply(frame_rgb, learningRate=0)
        self._foreground_msk = cv2.morphologyEx((255 * (temp_msk == 255)).astype(np.uint8), cv2.MORPH_OPEN, self._MORPH_OPEN_KERNEL)
        self._shadow_msk = cv2.morphologyEx((255 * (temp_msk == 127)).astype(np.uint8), cv2.MORPH_OPEN, self._MORPH_OPEN_KERNEL)
        self._drop_msk = cv2.morphologyEx((255 * np.logical_and(frame_gray <= 50, np.logical_or(self._foreground_msk, self._shadow_msk))).astype(np.uint8), cv2.MORPH_OPEN, self._MORPH_OPEN_KERNEL)
        # Detect drops.
        keypoints = self._blob_detector.detect(self._drop_msk)
        drops = list()
        for keypoint in keypoints:
            half_size = keypoint.size / 2
            bbox = BoundingBox(keypoint.pt[0] - half_size, keypoint.pt[1] - half_size, keypoint.size, keypoint.size)
            may_be_new = self._may_be_new(keypoint.pt)
            drops.append((bbox, may_be_new))
        return drops

    @property
    def drop_msk(self) -> np.ndarray:
        return self._drop_msk

    @property
    def foreground_msk(self) -> np.ndarray:
        return self._foreground_msk

    @property
    def shadow_msk(self) -> np.ndarray:
        return self._shadow_msk
