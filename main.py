import torch.nn as nn
import torch
import math


class Model(nn.Module):
    def __init__(self, npoint):
        super().__init__()
        self.npoint = npoint
        coords = torch.Tensor(npoint, 2)
        self.coords = coords
        
        # initialize coords
        nn.init.kaiming_uniform_(self.coords, a=math.sqrt(5))
    
    def forward(self):
        return self.coords



def loss_func(coords, gt):
    # coords.shape = [npoint, 2]
    # gt is the ground truth pair distances
    # gt.shape = [npoint, npoint]
    # the idea is to transpose coord, broadcast both, and substract
    npoint = coords.shape[0]
    copy1 = torch.reshape(coords, (1, npoint, 2))
    copy2 = torch.reshape(coords, (npoint, 1, 2))
    broadcasted1 = torch.broadcast_to(copy1, (npoint, npoint, 2))
    broadcasted2 = torch.broadcast_to(copy2, (npoint, npoint, 2))
    diff = broadcasted1 - broadcasted2 # [npoint, npoint, 2]
    diff_sq = torch.square(diff) # [npoint, npoint, 2]
    pred_dist = torch.sqrt(torch.sum(diff_sq, dim=2))
    pred_dist = torch.reshape(pred_dist, (npoint, npoint))# [npoint, npoint]
    mask = gt.gt(0)
    pred_dist = torch.where(mask, pred_dist, gt)
    err = torch.abs(pred_dist - gt) / gt
    return torch.sum(err)


gt = torch.tensor([[0,2,0,0,0],[2,0,3,0,0],[0,3,0,2,2],[0,0,2,0,2],[0,0,2,2,0]], dtype=torch.float)

model = Model(5)
output = model()
loss = loss_func(output, gt)
loss.requires_grad = True
loss.backward()
print(model.coords.grad)
