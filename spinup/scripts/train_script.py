import pdb, argparse, os
import torch
import torch.nn as nn

import gym
from gym import wrappers
# from TTenv import Display2D
#Algs
from spinup import sqn_pytorch
#Agents
from race_agents.ego_agent.agents import PurePursuitAgent as EgoPurePursuit
from race_agents.opp_agent.agents import PurePursuitAgent as OppPurePursuit

BASE_DIR = os.path.dirname('/'.join(str.split(os.path.realpath(__file__),'/')[:-2]))

def main():
    # making the environment
    racecar_env = gym.make('f110_gym:f110-RL-v0')

    # Initial state for ego and opp
    initialization = {}
    initialization['initial_x'] = [0.0, 2.0]
    initialization['initial_y'] = [0.0, 0.0]
    initialization['initial_theta'] = [0.0, 0.0]
    lap_time = 0.0

    # Params for env
    wheelbase = 0.3302
    mass= 3.74
    l_r = 0.17145
    I_z = 0.04712
    mu = 0.523
    h_cg = 0.074
    cs_f = 4.718
    cs_r = 5.4562
    exec_dir = BASE_DIR + '/f1tenth_gym/build/'
    map_path = BASE_DIR + '/f1tenth_gym/maps/skirk.yaml'
    map_img_ext = '.png'

    #Params for ego agent
    path_nums = [3,4,5,6,7]
    ego_csv_paths = []
    for num in path_nums:
        ego_csv_paths.append(BASE_DIR + '/race_agents/ego_agent/waypoints/Multi-Paths/multiwp%d.csv'%(num))
    ego_agent = EgoPurePursuit(ego_csv_paths, wheelbase)

    #Params for opponent agent
    opp_csv_path = BASE_DIR + '/race_agents/opp_agent/skirk.csv'
    opp_agent = OppPurePursuit(opp_csv_path, wheelbase)

    # init gym backend
    racecar_env.init_map(map_path, map_img_ext, False, False)
    racecar_env.update_params(mu, h_cg, l_r, cs_f, cs_r, I_z, mass, exec_dir, double_finish=True)

    exp_name = 'tests'

    # Wrappers
    # env = Display2D(env)

    # Create env function
    env_fn = lambda : racecar_env
    
    #Training function
    ac_kwargs = dict(hidden_sizes=[64,64], activation=nn.ReLU)
    logger_kwargs = dict(output_dir='data/sqn/'+exp_name, exp_name=exp_name)

    sqn_pytorch(env_fn=env_fn, env_init=initialization, ego_agent=ego_agent, opp_agent=opp_agent, 
        ac_kwargs=ac_kwargs, steps_per_epoch=5000, epochs=args.epochs, 
        logger_kwargs=logger_kwargs, save_freq=args.checkpoint_freq, 
        polyak=args.polyak, alpha=args.alpha, batch_size=args.batch)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--checkpoint_freq', type=int, default=2)
    parser.add_argument('--polyak', type=float, default=0.995)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--batch', type=int, default=100)
    args = parser.parse_args()
    
    main()

""" Environment example usage
    Resetting the environment
    
    obs, step_reward, done, info = racecar_env.reset({'x': initial_x,
                                                      'y': initial_y,
                                                      'theta': initial_theta})
    # Simulation loop
    while not done:

        # Your agent here
        ego_speed, opp_speed, ego_steer, opp_steer = agent.plan(obs)

        # Stepping through the environment
        action = {'ego_idx': 0, 'speed': [ego_speed, opp_speed], 'steer': [ego_steer, opp_steer]}
        obs, step_reward, done, info = racecar_env.step(action)

        # Getting the lap time
        lap_time += step_reward
"""