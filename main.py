import torch.nn as nn
import torch
import math
import numpy as np


class Model(nn.Module):
    def __init__(self, npoint):
        super().__init__()
        self.npoint = npoint
        coords = torch.Tensor(npoint, 2)
        self.coords = nn.Parameter(data=coords, requires_grad=True)
        
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
    # Hmmmm sqrt is not differentiable
    # pred_dist = torch.sqrt(torch.sum(diff_sq, dim=2).reshape((npoint, npoint)))
    pred_dist = torch.sum(diff_sq, dim=2).reshape((npoint, npoint))
    mask = gt.gt(0)
    pred_dist = torch.where(mask, pred_dist, gt)
    gt = torch.square(gt)
    err = torch.abs(pred_dist - gt)
    err = torch.where(mask, err / gt, gt)
    loss = torch.sum(err)
    print(loss)
    return loss


gt = torch.tensor([[0,2,0,0,0],[2,0,3,0,0],[0,3,0,2,2],[0,0,2,0,2],[0,0,2,2,0]], dtype=torch.float)
model = Model(5)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
model.train()

# just for logging
steps = 200
coord_changes = np.zeros((steps, 5, 2), dtype=np.float32)

for step in range(steps):
    # print(model.coords)
    optimizer.zero_grad()
    output = model()
    loss = loss_func(output, gt)
    loss.backward()
    # print(model.coords.grad)
    optimizer.step()

    coord = model.coords.cpu().detach().numpy()
    coord_changes[step] = coord
    # print(f"({coord[0][0]}, {coord[0][1]})")

with open('coord_changes.bin', 'wb') as f:
    coord_changes.tofile(f)