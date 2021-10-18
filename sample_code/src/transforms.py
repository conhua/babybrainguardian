import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ToTensor:

    def __call__(self, x):
        return torch.from_numpy(x).type(torch.float)


class RandomResizedCrop(nn.Module):
    def __init__(self, output_size, scale):
        super(RandomResizedCrop, self).__init__()
        self.output_size = output_size
        self.scale = scale

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).type(torch.float)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        length = x.size(-1)
        crop_size = int(length * self.scale)
        index = torch.randint(0, length - crop_size + 1, size=(1,)).item()
        x = x[..., :, index: index + crop_size]
        x = F.interpolate(x, self.output_size, mode="linear", align_corners=False).squeeze()
        return x


class TwoCropTransform:

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class Compose:

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x
