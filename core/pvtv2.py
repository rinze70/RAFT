from turtle import forward
import torch
import torch.nn as nn
import timm
import numpy as np

class pvt_v2(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.pvt = timm.create_model('pvt_v2_b5', pretrained=pretrained)

        del self.pvt.head
        del self.pvt.stages[2]
        del self.pvt.stages[2]

    def forward(self, x):
        x, feat_size = self.pvt.patch_embed(x)
        for stage in self.pvt.stages:
            x, feat_size = stage(x, feat_size=feat_size)
        return x
        
    def compute_params(self):
        num = 0

        for stage in self.pvt.stages:
            for param in stage.parameters():
                num +=  np.prod(param.size())
        
        return num

if __name__ == '__main__':
    m = pvt_v2()
    print(m.compute_params())
    # m = timm.create_model('pvt_v2_b0', pretrained=True)
    input = torch.randn(2, 3, 400, 800)
    out = m(input)
    # out = m.forward_features(input)
    print(out.shape)