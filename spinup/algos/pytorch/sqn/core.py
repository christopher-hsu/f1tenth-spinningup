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

def process_obs(obs):

    '''
        Args: 
            obs: observation that we get out of each step
        Returns:
            obs_array: 15 features extracted that include: ego_pose,opp_pose, 3 closest points using lidar. All of the
            features are in global frame (numpy array shape (16,))
    '''

    num_smallest = 3

    obs_array = np.zeros((10 + (num_smallest*2),))

    obs_array[0] = obs['poses_x'][0] # ego pos_x
    obs_array[1] = obs['poses_y'][0] # ego pos_y
    obs_array[2] = obs['poses_theta'][0] # ego theta

    R_mat_ego = np.array([[np.cos(obs_array[2]),np.sin(obs_array[2])],
                        [-np.sin(obs_array[2]),np.cos(obs_array[2])]])

    ego_vel_car_frame = np.zeros((1,2))

    ego_vel_car_frame[0,0] = obs['linear_vels_x'][0]
    ego_vel_car_frame[0,1] = 0

    ego_vel_global = np.dot(ego_vel_car_frame,R_mat_ego)

    obs_array[3:5] = ego_vel_global[0,:] # ego velocity x and y

    obs_array[5] = obs['poses_x'][1] # opp pos_x
    obs_array[6] = obs['poses_y'][1] # opp pos_y
    obs_array[7] = obs['poses_theta'][1] # opp theta

    R_mat_opp = np.array([[np.cos(obs_array[7]),np.sin(obs_array[7])],
                        [-np.sin(obs_array[7]),np.cos(obs_array[7])]])

    opp_vel_car_frame = np.zeros((1,2))

    opp_vel_car_frame[0,0] = obs['linear_vels_x'][1]

    opp_vel_car_frame[0,1] = 0

    opp_vel_global = np.dot(ego_vel_car_frame,R_mat_opp)

    obs_array[8:10] = opp_vel_global[0,:] # opp velocity x and y

    ### Process the lidar data to get the postion of the closest 3 points:

    ranges = np.array(list(obs['scans'][0]))
    angles = np.linspace(-4.7/2., 4.7/2. num=ranges.shape[0])

    ranges = ranges[np.isfinite(ranges)]
    angles = angles[np.isfinite(ranges)]

    k_smallest_idx = np.argpartition(ranges,num_smallest)
    k_smallest_idx = k_smallest_idx[:num_smallest]

    lidar_xy_car_frame = np.zeros((num_smallest,2))

    lidar_xy_car_frame[:,0] = (ranges[k_smallest_idx] * np.cos(angles[k_smallest_idx]))
    lidar_xy_car_frame[:,1] = (ranges[k_smallest_idx] * np.sin(angles[k_smallest_idx]))

    lidar_xy_global = np.dot(lidar_xy_car_frame,R_mat_ego)

    obs_array[10:] = lidar_xy_global.flatten()

    return obs_array

class MLPActionSelector(nn.Module):

    def __init__(self, alpha):
        super().__init__()
        self.log_softmax_batch = nn.LogSoftmax(dim=1) 
        self.log_softmax = nn.LogSoftmax(dim=0) 
        self.alpha = alpha

    def action(self, q, deterministic=False, with_logprob=True):

        # pi_log = self.log_softmax(q/self.alpha)
        soft_q = q/self.alpha

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
