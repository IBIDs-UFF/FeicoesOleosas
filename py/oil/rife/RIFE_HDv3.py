from .IFNet_HDv3 import *
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import torch


class Model(torch.nn.Module):

    def __init__(self, local_rank=-1):
        super(Model, self).__init__()
        self.flownet = IFNet()
        if local_rank != -1:
            self.flownet = DDP(self.flownet, device_ids=[local_rank], output_device=local_rank)

    def load_model(self, path, rank=0):
        def convert(param):
            if rank == -1:
                return {k.replace('module.', ''): v for k, v in param.items() if 'module.' in k}
            else:
                return param
        if rank <= 0:
            if torch.cuda.is_available():
                self.flownet.load_state_dict(convert(torch.load(os.path.join(path, 'flownet.pkl'))))
            else:
                self.flownet.load_state_dict(convert(torch.load(os.path.join(path, 'flownet.pkl'), map_location='cpu')))

    def forward(self, img0, img1, scale=1.0):
        scale_list = [4/scale, 2/scale, 1/scale]
        return self.flownet(img0, img1, scale_list)
