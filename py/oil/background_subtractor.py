from torch import Tensor
import torch


class BackgroundSubtractor():

    def __init__(self, samples_lab: Tensor) -> None:
        samples_lab = samples_lab.permute(0, 2, 3, 1).reshape(-1, 3)
        self._mean = torch.mean(samples_lab, axis=0)
        self._inv_cov = torch.linalg.inv(torch.cov((samples_lab - self._mean).T))

    def apply(self, img_lab: Tensor) -> Tensor:
        _, height, width = img_lab.shape
        return (torch.matmul(img_lab.permute(1, 2, 0).reshape(-1, 3) - self._mean, self._inv_cov) > 2.5).any(axis=1).view(height, width)
