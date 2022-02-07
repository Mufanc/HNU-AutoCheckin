import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

from .splitter import split
from .configs import *

pipeline = transforms.Compose([
    transforms.ToTensor(),  # 将图片转换成 Tensor
    transforms.Normalize((0.1307,), (0.3081,)),  # 正则化：降低模型复杂度（防止过拟合）
])


class Captcha(Dataset):
    def __init__(self, image):
        self.digits = split(image)
        if self.digits is None:
            raise ValueError('Not enough digits to unpack!')

    def __getitem__(self, index):
        return pipeline(self.digits[index].astype(np.float32))

    def __len__(self):
        return DIGITS
