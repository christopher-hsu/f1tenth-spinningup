import pdb, argparse
# import tensorflow as tf
import torch
import torch.nn as nn

import gym
from gym import wrappers
# from TTenv import Display2D
#Algs
# from spinup import ppo
from spinup import sqn_pytorch

#Envs
# from TTenv import MultiGoalCont
# from TTenv import RandGoalCont
# from TTenv import NoBelief
# from TTenv import TargetTrackingContEnv0
# from TTenv import TargetTrackingContEnv1


def main():
    # making the environment
    racecar_env = gym.make('f110_gym:f110-RL-v0')

    # Initial state
    initialization = {}
    initialization['initial_x'] = [0.0, 2.0]
    initialization['initial_y'] = [0.0, 0.0]
    initialization['initial_theta'] = [0.0, 0.0]
    lap_time = 0.0

    # wheelbase = 0.3302
    mass= 3.74
    l_r = 0.17145
    I_z = 0.04712
    mu = 0.523
    h_cg = 0.074
    cs_f = 4.718
    cs_r = 5.4562
    exec_dir = '/home/chsu/repositories/f1tenth-spinningup/f1tenth_gym/build/'
    map_path = '../f1tenth_gym/maps/skirk.yaml'
    map_img_ext = '.png'

    # init gym backend
    racecar_env.init_map(map_path, map_img_ext, False, False)
    racecar_env.update_params(mu, h_cg, l_r, cs_f, cs_r, I_z, mass, exec_dir, double_finish=True)

    # Resetting the environment
    # obs, step_reward, done, info = racecar_env.reset({'x': initial_x,
                                                      # 'y': initial_y,
                                                      # 'theta': initial_theta})
    # import pdb;pdb.set_trace()
    # # Simulation loop
    # while not done:

    #     # Your agent here
    #     ego_speed, opp_speed, ego_steer, opp_steer = agent.plan(obs)

    #     # Stepping through the environment
    #     action = {'ego_idx': 0, 'speed': [ego_speed, opp_speed], 'steer': [ego_steer, opp_steer]}
    #     obs, step_reward, done, info = racecar_env.step(action)

    #     # Getting the lap time
    #     lap_time += step_reward

    exp_name = 'tests'

    # # Wrappers
    # env = wrappers.TimeLimit(env, max_episode_steps=args.horizon)
    # env = Display2D(env)
    # # Create env function, future use gym.make()
    env_fn = lambda : racecar_env
    
    #Training function
    ac_kwargs = dict(hidden_sizes=[64,64], activation=nn.ReLU)
    logger_kwargs = dict(output_dir='data/sac/'+exp_name, exp_name=exp_name)

    sqn_pytorch(env_fn=env_fn, env_init=initialization, ac_kwargs=ac_kwargs, steps_per_epoch=5000, 
        epochs=args.epochs, logger_kwargs=logger_kwargs, save_freq=args.checkpoint_freq)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('--algos', help='agent policy', default='sac')
    # parser.add_argument('--env', help='environment ID', default='MultiGoalCont')
    # parser.add_argument('--map_name', help='map ID', default='empty')
    # parser.add_argument('--metaenv', default=None)
    # parser.add_argument('--horizon', type=int, default=150)
    # parser.add_argument('--num_targets', type=int, default=2)
    # parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--checkpoint_freq', type=int, default=2)
    parser.add_argument('--is_training', type=bool, default=True)
    args = parser.parse_args()
    
    main()