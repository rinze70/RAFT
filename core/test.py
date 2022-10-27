import torch

from raft import RAFT
from corr import CorrBlock, QuadTreeCorrBlock

if __name__ == '__main__':
    device = torch.device('cuda')
    image1 = torch.rand(2, 3, 384, 512)

    fmap1 = torch.rand(2, 128, 384//8, 512//8).to(device)
    fmap2 = torch.rand(2, 128, 384//8, 512//8).to(device)

    corr_fn = QuadTreeCorrBlock(fmap1, fmap2, topks=[16, 8, 8, 8], radius=3)

    coords0, coords1 = RAFT.initialize_flow(image1)

    corr = corr_fn(coords1)

    for l in corr:
        print(l.shape)