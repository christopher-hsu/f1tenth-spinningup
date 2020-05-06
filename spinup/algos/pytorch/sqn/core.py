import numpy as np
import scipy.signal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


class MLPActionSelector(nn.Module):
    '''
    Soft parameterization of q value logits,
    pi_log = (1/Z)*(e^((v(x)/alpha) - min((v(x)/alpha)))
    If determinstic take max value as action,
    Else (stochastic),
    Sample from multinomial of the soft logits.
    '''
    def __init__(self, alpha, act_dim):
        super().__init__()
        self.alpha = alpha
        self.act_dim = act_dim

        # self.softmax = nn.Softmax(dim=1)
        self.logsoftmax = nn.LogSoftmax(dim=1)


    def forward(self, q, action_mask, deterministic=False, with_logprob=True):
        #Divide by temperature term, alpha
        q_soft = q/self.alpha

        # Mask out actions not available
        mask = np.ones(self.act_dim, dtype=bool)  #16 paths + optimal path
        mask[list(action_mask)] = False
        try:
            q_soft[:, mask] = -float("Inf")
        except:
            q_soft = q_soft.unsqueeze(0)
            q_soft[:, mask] = -float("Inf")

        pi_log = self.logsoftmax(q_soft)

        if deterministic:
            mu = torch.argmax(pi_log)
            pi_action = mu      
        else:
            try:
                q_log_dist = torch.distributions.multinomial.Multinomial(1, logits=pi_log)
                action = q_log_dist.sample()
                pi_action = torch.argmax(action, dim=1, keepdim=True)

            except: #This case happens if no paths are available -> 0.5 vel and 0 steer, force it crash and learn
                pi_action = torch.argmax(pi_log, dim=1, keepdim=True)   
                pi_action = (torch.ones([pi_log.shape[0],1]) * 15).type(torch.long)

        if with_logprob:
            logp_pi = torch.gather(pi_log,1,pi_action)
        else:
            logp_pi = None
        
        return pi_action, logp_pi

class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.vf_mlp = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    # v(x)
    def values(self, obs):
        v_x = self.vf_mlp(obs)
        return v_x

    # q(x,a)
    def forward(self, obs, act):
        v_x = self.vf_mlp(obs)
        q = torch.gather(v_x, 1, act.type(torch.LongTensor))

        return q


class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, alpha, hidden_sizes=(256,256),
                 activation=nn.ReLU):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.n

        # build policy and value functions
        self.pi = MLPActionSelector(alpha, act_dim)
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(obs_dim, act_dim,  hidden_sizes, activation)

    def act(self, obs, action_mask, deterministic=False):
        with torch.no_grad():
            v1 = self.q1.values(obs)
            v2 = self.q2.values(obs)

            a, _ = self.pi(v1+v2, action_mask, deterministic, False)
            # Tensor to int
            return int(a)
