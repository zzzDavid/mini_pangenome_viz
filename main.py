import torch.nn as nn
import torch
import math
import numpy as np


class Model(nn.Module):
    def __init__(self, npoint, init=None):
        super().__init__()
        self.npoint = npoint
        if init is None:
            coords = torch.Tensor(npoint, 2)
            self.coords = nn.Parameter(data=coords, requires_grad=True)
            
            # initialize coords
            nn.init.kaiming_uniform_(self.coords, a=math.sqrt(5))
        else:
            self.coords = nn.Parameter(data=init, requires_grad=True)
    
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
    pred_dist = torch.norm(diff, dim=2).reshape((npoint, npoint))
    mask = gt.gt(0)
    pred_dist = torch.where(mask, pred_dist, gt)
    gt = torch.square(gt)
    err = torch.abs(pred_dist - gt)
    err = torch.where(mask, err / gt, gt)
    loss = torch.sum(err)
    return loss

def generate_init(gt):
    # gt.shape = [npoint, npoint]
    # generate initial coords
    npoint = gt.shape[0]
    coords = torch.zeros((npoint, 2))
    for i in range(npoint):
        dist = gt[0, i]
        if dist == 0:
            continue
        else:
            # generate random angle
            angle = torch.rand(1) * 2 * math.pi
            coords[i, 0] = dist * math.cos(angle)
            coords[i, 1] = dist * math.sin(angle)
    return coords

filename = './preprocess/distance.bin'
npoint = 9898
print("loading ground truth distance matrix: {}...".format(filename))
gt = np.fromfile(filename, dtype=np.int32).reshape((npoint, npoint))
gt = torch.from_numpy(gt)
# init = generate_init(gt)
init = None
model = Model(npoint, init)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
# Use adam instead of SGD
optimizer = torch.optim.Adam(model.parameters(), lr=0.2)
model.to('cuda')
gt = gt.to('cuda')
model.train()

# just for logging
steps = 100000
log_interval = 500
coord_changes = np.zeros((steps//log_interval, npoint, 2), dtype=np.float32)

for step in range(steps):
    # print(model.coords)
    optimizer.zero_grad()
    output = model()
    loss = loss_func(output, gt)
    loss.backward()
    # print(model.coords.grad)
    optimizer.step()
    if step % log_interval == 0:
        print("step {}, loss {}".format(step, loss))
    # scheduler.step()

        coord = model.coords.cpu().detach().numpy()
        # snapshot the coords
        filename = "./snapshots/coords_{}.bin".format(step)
        coord.tofile(filename)
        coord_changes[step//log_interval] = coord
    # print(f"({coord[0][0]}, {coord[0][1]})")

with open('coord_changes.bin', 'wb') as f:
    coord_changes.tofile(f)