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


LOG_STD_MAX = 2
LOG_STD_MIN = -20

class SquashedGaussianMLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):#, act_limit):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        # self.act_limit = act_limit

    def forward(self, obs, deterministic=False, with_logprob=True):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
            # pi_action = torch.argmax(mu,dim=1)
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding 
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290) 
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        # pi_action = self.act_limit * pi_action

        return pi_action, logp_pi

class MLPActionSelector(nn.Module):

    def __init__(self, alpha):
        super().__init__()
        self.log_softmax_batch = nn.LogSoftmax(dim=1) 
        self.log_softmax = nn.LogSoftmax(dim=0) 
        self.alpha = alpha

    def action(self, q, deterministic=False, with_logprob=True):

        pi_log = self.log_softmax(q/self.alpha)

        if deterministic:
            mu = torch.argmax(pi_log)
            pi_action = mu      
        else:
            #not sure about the negative
            pi_action = torch.multinomial(-pi_log,1)

        if with_logprob:
            logp_pi = torch.gather(pi_log,1,pi_action)
            pi_action = pi
        else:
            logp_pi = None
        
        return pi_action, logp_pi


    def forward(self, q, deterministic=False, with_logprob=True):

        pi_log = self.log_softmax_batch(q/self.alpha)

        if deterministic:
            mu = torch.argmax(pi_log, dim=1, keepdim=True)
            pi_action = mu      
        else:
            #not sure about the negative
            pi = torch.multinomial(-pi_log,1)

        if with_logprob:
            logp_pi = torch.gather(pi_log,1,pi)
            pi_action = pi
        else:
            logp_pi = None
        
        return pi_action, logp_pi


class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        # self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)
        self.vf_mlp = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def values(self, obs):
        v_x = self.vf_mlp(obs)

        return v_x

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
        # self.pi = SquashedGaussianMLPActor(obs_dim, act_dim, hidden_sizes, activation) #, act_limit)
        self.pi = MLPActionSelector(alpha)
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(obs_dim, act_dim,  hidden_sizes, activation)

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            v1 = self.q1.values(obs)
            v2 = self.q2.values(obs)
            a, _ = self.pi.action(v1+v2, deterministic, False)
            return a.numpy()
