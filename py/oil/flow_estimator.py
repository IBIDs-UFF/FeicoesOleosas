from .rife.RIFE_HDv3 import Model
from torch import Tensor
from typing import Tuple
import os
import torch


Device = torch.device


class FlowEstimator:

    @torch.no_grad()
    def __init__(self, device: Device) -> None:
        # Setup the RIFE model.
        self._model = Model()
        self._model.load_model(os.path.join(os.path.dirname(__file__), 'rife'), -1)
        self._model.eval()
        self._model.to(device)

    @torch.no_grad()
    def apply(self, img0: Tensor, img1: Tensor, scale: float) -> Tuple[Tensor, Tensor]:
        flow, mask = self._model(img0, img1, scale)
        return flow, mask
