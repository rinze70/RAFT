import torch

from raft import RAFT
from corr import CorrBlock

if __name__ == '__main__':
    image1 = torch.rand(2, 3, 384, 512)

    fmap1 = torch.rand(2, 128, 384//8, 512//8)
    fmap2 = torch.rand(2, 128, 384//8, 512//8)

    corr_fn = CorrBlock(fmap1, fmap2, radius=3)

    coords0, coords1 = RAFT.initialize_flow(image1)

    corr = corr_fn(coords1)