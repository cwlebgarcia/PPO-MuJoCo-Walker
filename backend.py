import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np

class ValueFunction(nn.Module):
    def __init__(self, xdim, hdim=64):
        super().__init__()
        self.v_net = nn.Sequential(
                    nn.Linear(xdim, hdim),
                    nn.Tanh(),
                    nn.Linear(hdim, hdim),
                    nn.Tanh(),
                    nn.Linear(hdim, 1),
                  )
        
    def forward(s,x):
        return torch.squeeze(s.v_net(x),-1)
    
class uth_t(nn.Module):
    def __init__(s,xdim,udim,
                 hdim=64,fixed_var = True):
        super().__init__()
        s.xdim,s.udim = xdim, udim
        s.fixed_var = fixed_var

        ### TODO
        s.q_net = nn.Sequential(
                    nn.Linear(xdim, hdim),
                    nn.Tanh(),
                    nn.Linear(hdim, hdim),
                    nn.Tanh(),
                    nn.Linear(hdim, hdim),
                    nn.Tanh(),
                    nn.Linear(hdim, udim)
                  )
        s.log_std = torch.nn.Parameter(torch.as_tensor(-0.5 * np.ones(udim,dtype=np.float32)),requires_grad = False)
        ### END TODO

    def forward(s,x):
        ### TODO
        mu = s.q_net(x)
        std = torch.exp(s.log_std).expand_as(mu)
        ### END TODO
        return mu,std

class ActorCritic(nn.Module):
    def __init__(s, xdim, udim, hdim=32):
        super().__init__()
        
        s.pi = uth_t(xdim,udim,hdim)
        s.v  = ValueFunction(xdim,hdim)

    def step(s, x):
        
        pi = s.pi.distribution(x)
        a = pi.sample()
        logp_a = s.pi.logp(pi, a)
        v = s.v(x)
        return a.cpu().numpy(), v.cpu().numpy(), logp_a.cpu().numpy()

    def act(s, x):
        return s.step(x)[0]