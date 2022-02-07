# 加载必要的库
import os

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from .dataset import Captcha
from .configs import *


# 构建网络模型
class Digit(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 10, (3, 3))  # 1：图片通道数，10：输出通道数，5：卷积核大小
        self.fc1 = nn.Linear(1040, 300)  # 1040：输入通道，300：输出通道
        self.fc2 = nn.Linear(300, 10)

    def forward(self, x):
        # 卷积层
        input_size = x.size(0)
        x = F.relu(self.conv(x))  # 输入：batch*1*10*15，输出：batch*10*8*13
        x = x.view(input_size, -1)  # 拉伸，-1：自动计算维度

        # 全连接层 1
        x = F.relu(self.fc1(x))  # 输入：batch*1040，输出：batch*300

        # 全连接层 2
        x = self.fc2(x)  # 输入：batch*300，输出：batch*10

        return F.log_softmax(x, dim=1)  # 计算分类后，每个数字的概率值


def recognize(image):
    loader = DataLoader(Captcha(image), batch_size=DIGITS)
    with torch.no_grad():
        data = iter(loader).__next__().to('cpu')
        model = Digit()
        model.load_state_dict(torch.load(os.path.join(
            os.path.split(__file__)[0],
            'model.pck'
        )))
        return ''.join([f'{int(x.argmax())}' for x in model(data)])
