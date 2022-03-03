from .background_subtractor import BackgroundSubtractor
from torch import Tensor
from typing import List, NamedTuple, Tuple
import cv2
import kornia
import math
import numpy as np
import torch


class BoundingBox(NamedTuple):
    x: float
    y: float
    w: float
    h: float


class Circle(NamedTuple):
    center: np.ndarray
    radius: float


class DropDetector():

    _MORPH_OPEN_KERNEL = torch.ones((5, 5), dtype=torch.float32)

    def __init__(self, subtractor: BackgroundSubtractor) -> None:
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
        self._subtractor = subtractor
        self._circles: List[Circle] = list()
        self._is_background: Tensor = None
        self._is_drop: Tensor = None

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

    def detect(self, frame_rgb: Tensor, frame_gray: Tensor) -> List[Tuple[BoundingBox, bool]]:
        self._MORPH_OPEN_KERNEL = self._MORPH_OPEN_KERNEL.to(frame_rgb.device)
        # Compute birth regions.
        height, width = frame_gray.shape
        radius = 0.10 * min(width, height)
        self._circles.clear()
        for _ in range(10):
            self._circles.append((np.random.rand(2), np.random.rand()))
        #TODO Substituir
        # lines = kht(cv2.Canny(frame_gray, 80, 200))
        # for line_ind1 in range(3):
        #     line1 = self._compute_line_coefficients(lines[line_ind1], (width, height))
        #     for line_ind2 in range(line_ind1 + 1, 3):
        #         line2 = self._compute_line_coefficients(lines[line_ind2], (width, height))
        #         pt = np.cross(line1, line2)
        #         if pt[-1] != 0:
        #             pt = pt[:-1] / pt[-1]
        #             if 0 <= pt[0] < width and 0 <= pt[1] < height:
        #                 self._circles.append((pt, radius))
        # Compute foreground, shadow, and drop masks.
        is_foreground = self._subtractor.apply(kornia.color.rgb_to_lab(frame_rgb))
        self._is_background = kornia.morphology.opening(torch.logical_not(is_foreground).to(torch.float32).view(1, 1, height, width), self._MORPH_OPEN_KERNEL).view(height, width).to(torch.bool)
        self._is_drop = kornia.morphology.opening(torch.logical_and(frame_gray <= 50, is_foreground).to(torch.float32).view(1, 1, height, width), self._MORPH_OPEN_KERNEL).view(height, width).to(torch.bool)
        # Detect drops.
        keypoints = self._blob_detector.detect(kornia.tensor_to_image((255 * self._is_drop).to(torch.uint8)))
        drops = list()
        for keypoint in keypoints:
            half_size = keypoint.size / 2
            bbox = BoundingBox(keypoint.pt[0] - half_size, keypoint.pt[1] - half_size, keypoint.size, keypoint.size)
            may_be_new = self._may_be_new(keypoint.pt)
            drops.append((bbox, may_be_new))
        # Return detected drops.
        return drops

    @property
    def is_background(self) -> Tensor:
        return self._is_background
