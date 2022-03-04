from .rife.RIFE_HDv3 import Model
from torch import Tensor
import os
import torch


class FlowEstimator:

    @torch.no_grad()
    def __init__(self, *, device: torch.device, scale: float, used_height: int, used_width: int) -> None:
        # Setup the RIFE model.
        self._model = Model()
        self._model.load_model(os.path.join(os.path.dirname(__file__), 'rife'), -1)
        self._model.eval()
        self._model.to(device)
        # Initialize other attributes.
        self._scale = scale
        self._used_width = used_width
        self._used_height = used_height

    @torch.no_grad()
    def estimate(self, padded_img0: Tensor, padded_img1: Tensor) -> Tensor:
        flow, mask = self._model(padded_img0, padded_img1, self._scale)
        mask = mask[0, 0, :self._used_height, :self._used_width]
        flow = (flow[0, :2, :self._used_height, :self._used_width] * mask - flow[0, 2:, :self._used_height, :self._used_width] * (1 - mask)).permute(1, 2, 0)
        return flow
