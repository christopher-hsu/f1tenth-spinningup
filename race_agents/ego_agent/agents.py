import numpy as np
import csv
from race_agents.opp_agent.agent_utils import get_actuation, nearest_point_on_trajectory_py2, first_point_on_trajectory_intersecting_circle

class Agent(object):
    def __init__(self, csv_path):
        # TODO: load waypoints from csv
        self.waypoints = None
        self.safe_speed = 0.5
        
    def plan(self, obs):
        pass


class PurePursuitAgent(Agent):
    # Pure pursuit control to a specified lane
    def __init__(self, csv_path, wheelbase):
        super(PurePursuitAgent, self).__init__(csv_path)
        self.lookahead_distance = 1.0
        self.wheelbase = wheelbase
        self.max_reacquire = 10.

        self.waypoints = []
        for paths in csv_path:
            self.waypoints.append(np.loadtxt(paths, ndmin=2,delimiter=','))
 
    def _get_current_waypoint(self, waypoints, lookahead_distance, position, theta):
        wpts = waypoints
        nearest_point, nearest_dist, t, i = nearest_point_on_trajectory_py2(position, wpts)
        if nearest_dist < lookahead_distance:
            lookahead_point, i2, t2 = first_point_on_trajectory_intersecting_circle(position, lookahead_distance, wpts, i+t, wrap=True)
            if i2 == None:
                return None
            current_waypoint = np.empty(3) #[x ,y, speed]
            # x, y
            current_waypoint[:2] = waypoints[i2, :]
            # speed
            current_waypoint[2] = 4.5
            return current_waypoint
        elif nearest_dist < self.max_reacquire:
            return waypoints[i, :]
        else:
            return None

    def plan(self, obs, action):
        #Choose the path to follow
        path = self.waypoints[action]      
        # path = self.waypoints  
        
        pose_x = obs['poses_x'][1]
        pose_y = obs['poses_y'][1]
        pose_theta = obs['poses_theta'][1]
        position = np.array([pose_x, pose_y])
        lookahead_point = self._get_current_waypoint(path, self.lookahead_distance, position, pose_theta)
        if lookahead_point is None:
            return self.safe_speed, 0.0
        speed, steering_angle = get_actuation(pose_theta, lookahead_point, position, self.lookahead_distance, self.wheelbase)
        return speed, steering_angle