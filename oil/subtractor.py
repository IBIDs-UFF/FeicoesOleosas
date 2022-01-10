import numpy as np


class BackgroundSubtractor():

    def __init__(self) -> None:
        self._mean_lab: np.ndarray = None
        self._max_distance: float = None

    def apply(self, img_lab: np.ndarray) -> np.ndarray:
        height, width, _ = img_lab.shape
        return (np.matmul(img_lab.reshape(-1, 3) - self._mean, self._inv_cov) > 2.5).any(axis=1).reshape(height, width)
    
    def train(self, samples_lab: np.ndarray) -> None:
        samples_lab = samples_lab.reshape(-1, 3)
        self._mean = np.mean(samples_lab, axis=0)
        self._inv_cov = np.linalg.inv(np.cov((samples_lab - self._mean).T))
