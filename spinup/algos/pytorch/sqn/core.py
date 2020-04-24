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
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    #Used during execution on single observations
    def action(self, q, deterministic=False, with_logprob=True):
        #Divide by temperature term, alpha
        q_soft = q/self.alpha
        #Normalize to probabilties
        q_norm = q_soft - torch.min(q_soft)
        pi_log = torch.exp(q_norm)
        #Normalize
        pi_log = torch.div(pi_log,pi_log.sum())

        if deterministic:
            mu = torch.argmax(pi_log)
            pi_action = mu      
        else:
            pi_action = torch.multinomial(pi_log,1)

        if with_logprob:
            logp_pi = torch.gather(pi_log,1,pi_action)
        else:
            logp_pi = None
        
        return pi_action, logp_pi

    #Used during training on batches of obervations
    def forward(self, q, deterministic=False, with_logprob=True):
        #Divide by temperature term, alpha
        q_soft = q/self.alpha
        #Normalize to probabilties
        q_norm = q_soft - torch.min(q_soft)
        pi_log = torch.exp(q_norm)
        #Normalize
        pi_log = torch.div(pi_log,pi_log.sum(dim=1,keepdim=True))

        if deterministic:
            mu = torch.argmax(pi_log)
            pi_action = mu      
        else:
            pi_action = torch.multinomial(pi_log,1)

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
        self.pi = MLPActionSelector(alpha)
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(obs_dim, act_dim,  hidden_sizes, activation)

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            v1 = self.q1.values(obs)
            v2 = self.q2.values(obs)
            a, _ = self.pi.action(v1+v2, deterministic, False)
            #From tensor to np.array to scalar
            return np.asscalar(a.numpy())
