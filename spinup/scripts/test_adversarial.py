import pdb, argparse, json, os
import gym
from gym import wrappers
from race_agents.ego_agent.agents2 import PurePursuitAgent as EgoPurePursuit
from race_agents.opp_agent.agents2 import PurePursuitAgent as OppPurePursuit
from spinup.visualize.display_wrapper import Display2D
from spinup.utils.f1tenth_test_policy import load_pytorch_policy, run_adversarial_policy
BASE_DIR = os.path.dirname('/'.join(str.split(os.path.realpath(__file__),'/')[:-2]))

def main():
    # making the environment
    racecar_env = gym.make('f110_gym:f110-RL-v0')

    # Initial state for ego and opp
    initialization = {}
    initialization['initial_x'] = [[0.0, 2.0],[2.0, 0.0]]
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
    wheelbase2 = 0.3
    path_nums = list(range(2,18))
    ego_csv_paths = []
    for num in path_nums:
        ego_csv_paths.append(BASE_DIR + '/race_agents/ego_agent/waypoints/Multi-Paths2/multiwp%d.csv'%(num))
    ego_csv_paths.append(BASE_DIR + '/race_agents/ego_agent/waypoints/Multi-Paths2/multiwp-opt.csv') ## adding the opt path

    ego_agent = EgoPurePursuit(ego_csv_paths, wheelbase2)
    ego_action = load_pytorch_policy(args.path, itr=args.itr, deterministic=True)

    #Params for opponent agent
    opp_agent = OppPurePursuit(ego_csv_paths, wheelbase2)
    # opp_policy_paths = BASE_DIR + '/race_agents/opp_agent/opp_policy/rew-10_optimal'
    opp_policy_paths = BASE_DIR + '/race_agents/opp_agent/opp_policy/05051706'
    opp_action = load_pytorch_policy(opp_policy_paths, itr='', deterministic=True)

    # init gym backend
    racecar_env.init_map(map_path, map_img_ext, False, False)
    racecar_env.update_params(mu, h_cg, l_r, cs_f, cs_r, I_z, mass, exec_dir, double_finish=True)

    # Wrappers
    racecar_env = Display2D(racecar_env, map_path, map_img_ext, False, False)

    run_adversarial_policy(racecar_env, ego_action, opp_action, env_init=initialization, 
                    ego_agent=ego_agent, opp_agent=opp_agent, num_episodes=args.num_episodes, 
                    render=bool(args.render))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path', type=str, help='Path to config.json file')
    parser.add_argument('--itr', type=str, default='')
    parser.add_argument('--deterministic', type=bool, default=True)
    parser.add_argument('--num_episodes', type=int, default=100)
    parser.add_argument('--render', type=int, default=1)
    args = parser.parse_args()

    config_path = args.path + 'config.json'
    with open(config_path) as json_file:
        data = json.load(json_file)
    
    main()