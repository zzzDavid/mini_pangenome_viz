import torch.nn as nn


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
    # gt is the ground truth pair distances
    # gt.shape = [npoint, npoint]
    # the idea is to transpose coord, broadcast both, and substract
    pred_dist = 


