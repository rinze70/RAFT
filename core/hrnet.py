from turtle import forward
import torch
import torch.nn as nn
import timm
import numpy as np
from typing import List

class HRNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.hrn = timm.create_model('hrnet_w18_small', pretrained=pretrained, features_only=True)

        # del self.hrn.head
        # del self.hrn.stages[2]
        # del self.hrn.stages[2]

    def forward(self, x) -> List[torch.tensor]:
        out = []
        x = self.hrn.conv1(x)
        x = self.hrn.bn1(x)
        x = self.hrn.act1(x)
        if 0 in self.hrn._out_idx:
            out.append(x)
        x = self.hrn.conv2(x)
        x = self.hrn.bn2(x)
        x = self.hrn.act2(x)
        x = self.hrn.stages(x)
        if self.hrn.incre_modules is not None:
            x = [incre(f) for f, incre in zip(x, self.hrn.incre_modules)]
        for i, f in enumerate(x):
            if i + 1 in self.hrn._out_idx:
                out.append(f)
        return out
        
    def compute_params(self):
        num = 0

        for stage in self.hrn.stages:
            for param in stage.parameters():
                num +=  np.prod(param.size())
        
        return num

if __name__ == '__main__':
    m = HRNet()
    # print(m.compute_params())
    # m = timm.create_model('hrnet_w18', pretrained=True, features_only=True)
    input = torch.randn(2, 3, 368, 496) # 384 512; 368 496; 256 192
    out = m(input)
    # out = m.forward_features(input)
    print(out.shape)