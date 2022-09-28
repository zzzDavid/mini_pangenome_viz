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
            if isinstance(init, np.ndarray):
                init = init.astype(np.float32)
                init = torch.from_numpy(init)
            self.coords = nn.Parameter(data=init, requires_grad=True)
    
    def forward(self):
        return self.coords

# Path-distance-based loss func
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

# Energy based loss func
def energy_loss_func(coords, gt):
    # magnetic energy
    npoint = coords.shape[0]
    copy1 = torch.reshape(coords, (1, npoint, 2))
    copy2 = torch.reshape(coords, (npoint, 1, 2))
    broadcasted1 = torch.broadcast_to(copy1, (npoint, npoint, 2))
    broadcasted2 = torch.broadcast_to(copy2, (npoint, npoint, 2))
    diff = broadcasted1 - broadcasted2 # [npoint, npoint, 2]
    pairwise_dist = torch.norm(diff, dim=2).reshape(npoint, npoint)
    pairwise_dist = torch.abs(pairwise_dist - gt)
    return torch.sum(pairwise_dist) / torch.sum(gt) * 100

gt = np.fromfile('./preprocess/ground_truth.bin', dtype=np.int32)
npoint = int(np.sqrt(gt.shape[0])) # should be 3521
gt = gt.reshape((npoint, npoint)).astype(np.float32)
gt = torch.from_numpy(gt)
init = np.zeros((npoint, 2), dtype=np.float32) + 1
init_state = np.fromfile('./preprocess/spectral_init.bin', dtype=np.float32)
init_state =  init_state.reshape((-1, 2))
init[:init_state.shape[0], :] = init_state
init = init * 100
# init = None
model = Model(npoint, init)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
# Use adam instead of SGD
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
model.to('cuda')
gt = gt.to('cuda')
model.train()

# just for logging
steps = 100000
log_interval = 5000
coord_changes = np.zeros((steps//log_interval, npoint, 2), dtype=np.float32)

for step in range(steps):
    # print(model.coords)
    optimizer.zero_grad()
    output = model()
    # loss = loss_func(output, gt)
    loss = energy_loss_func(output, gt)
    loss.backward()
    # print(model.coords.grad)
    optimizer.step()
    if step % log_interval == 0:
        print("step {}, loss {}".format(step, loss))
        coord = model.coords.cpu().detach().numpy()
        # snapshot the coords
        # filename = "./snapshots/coords_{}.bin".format(step)
        # coord.tofile(filename)
        coord_changes[step//log_interval] = coord

with open('coord_changes.bin', 'wb') as f:
    coord_changes.tofile(f)