from .yolov5.models.common import DetectMultiBackend
from .yolov5.utils.general import check_img_size, non_max_suppression, scale_coords
from torch import Tensor
from typing import List, NamedTuple, Tuple
import kornia
import numpy as np
import os
import torch


YOLO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'yolov5'))
DROP_DETECTOR_WEIGHTS_FILEPATH = os.path.join(YOLO_DIR, 'drop_yolov5s.pt')


class BoundingBox(NamedTuple):
    x: float
    y: float
    w: float
    h: float


class DropDetector():

    @torch.no_grad()
    def __init__(self, *, confidence_threshold: float, device: torch.device, frame_size: int, iou_threshold: float) -> None:
        # Load model.
        self._model = DetectMultiBackend(DROP_DETECTOR_WEIGHTS_FILEPATH, device=device, dnn=False, data=os.path.join(YOLO_DIR, 'data', 'drop_data.yaml'))
        # Initialize other attributes.
        self._confidence_threshold = confidence_threshold
        self._img_size = check_img_size([frame_size, frame_size], s=self._model.stride)
        self._iou_threshold = iou_threshold
        # Warmup the model.
        if self._model.pt or self._model.jit:
            self._model.model.float()
        self._model.warmup(imgsz=(1, 3, *self._img_size), half=False)

    @torch.no_grad()
    def _resize_and_pad(self, im: Tensor) -> Tensor:
        # Input shape.
        _, height, width = im.shape
        # Scale ratio (new / old).
        r = min(self._img_size[0] / height, self._img_size[1] / width)
        # Compute padding.
        new_unpad_height, new_unpad_width = int(round(height * r)), int(round(width * r))
        dw, dh = self._img_size[1] - new_unpad_width, self._img_size[0] - new_unpad_height  # wh padding.
        if self._model.pt:  # Minimum rectangle.
            dw, dh = np.mod(dw, self._model.stride), np.mod(dh, self._model.stride)  # wh padding.
        dw /= 2  # Divide padding into 2 sides.
        dh /= 2
        if width != new_unpad_width or height != new_unpad_height:  # Resize.
            im = kornia.geometry.transform.resize(im, (new_unpad_height, new_unpad_width), interpolation='bilinear')
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        return torch.nn.functional.pad(im, (left, right, top, bottom), mode='constant', value=(114.0 / 255.0))

    @torch.no_grad()
    def detect(self, frame_rgb: Tensor) -> List[Tuple[BoundingBox, float]]:
        # Resize and pad image while meeting stride-multiple constraints.
        im = self._resize_and_pad(frame_rgb).unsqueeze(0)
        # Inference.
        pred = self._model(im, augment=False, visualize=False)
        pred = non_max_suppression(pred, conf_thres=self._confidence_threshold, iou_thres=self._iou_threshold, classes=None, agnostic=False, max_det=10)[0]
        # Process predictions.
        drops = list()
        if len(pred) > 0:
            # Rescale boxes to original size.
            pred[:, :4] = scale_coords(im.shape[2:], pred[:, :4], frame_rgb.shape[1:]).round()
            for x0, y0, x1, y1, confidence, _ in pred:
                bbox = BoundingBox(x0.item(), y0.item(), (x1 - x0).item(), (y1 - y0).item())
                drops.append((bbox, confidence.item()))
        # Return detected drops.
        return drops
