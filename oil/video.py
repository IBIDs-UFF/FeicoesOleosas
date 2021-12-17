from typing import Any, Optional, Tuple
import cv2
import numpy as np


class VideoReader:

    def __init__(self, filename: str) -> None:
        self._video = cv2.VideoCapture(filename)
        if not self._video.isOpened():
            raise RuntimeError('Error opening the video file')

    def __enter__(self) -> 'VideoReader':
        return self

    def __exit__(self, exc_type: Optional[Any], exc_val: Optional[Any], exc_tb: Optional[Any]) -> None:
        self._video.release()

    def __iter__(self) -> 'VideoReader':
        return self

    def __len__(self) -> int:
        return self.frame_count
    
    def __next__(self) -> np.ndarray:
        ret, frame = self._video.read()
        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            raise StopIteration

    def read(self) -> Optional[np.ndarray]:
        ret, frame_bgr = self._video.read()
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB) if ret else None
    
    @property
    def fps(self) -> int:
        return int(self._video.get(cv2.CAP_PROP_FPS))

    @property
    def frame_count(self) -> int:
        return int(self._video.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def height(self) -> int:
        return int(self._video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def num_channels(self) -> int:
        return 3

    @property
    def width(self) -> int:
        return int(self._video.get(cv2.CAP_PROP_FRAME_WIDTH))


class VideoWriter:

    def __init__(self, filename: str, size: Tuple[int, int], fps: int) -> None:
        self._video = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
        if not self._video.isOpened():
            raise RuntimeError('Error opening the video file')

    def __enter__(self) -> 'VideoWriter':
        return self

    def __exit__(self, exc_type: Optional[Any], exc_val: Optional[Any], exc_tb: Optional[Any]) -> None:
        self._video.release()

    def write(self, frame_rgb: np.ndarray) -> None:
        self._video.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
