import numpy as np
import csv
import math

class Agent(object):
    def __init__(self, csv_path):
        # TODO: load waypoints from csv
        self.safe_speed = 0.5
        
    def plan(self, obs):
        pass


class PurePursuitAgent(Agent):
    # Pure pursuit control to a specified lane
    def __init__(self, csv_path, wheelbase):
        super(PurePursuitAgent, self).__init__(csv_path)
        self.index = 0
        self.lookahead_distance = 0.85
        self.wheelbase = wheelbase
        self.waypoints = []
        for paths in csv_path:
            self.waypoints.append(np.loadtxt(paths, ndmin=2,delimiter=','))

 
    def _get_current_waypoint(self, waypoint, lookahead_distance, position, theta):

        R_mat_ego = np.array([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]])

        point_dist =  np.sqrt(np.sum(np.square(waypoint[:, 0:2]-position), axis=1))
        point_index = np.where(abs(point_dist-lookahead_distance)< 0.20)[0]
        #print(point_index)
        for index in point_index:
            l2_0 = [waypoint[index, 0]-position[0], waypoint[index,1]-position[1]]      #global frame
            goalx_veh = math.cos(theta)*l2_0[0] + math.sin(theta)*l2_0[1]     #local frame
            goaly_veh = -math.sin(theta)*l2_0[0] + math.cos(theta)*l2_0[1]    #lobal frame

            if abs(math.atan(goalx_veh/goaly_veh)) <  np.pi/2 and goalx_veh>0 :
                 return waypoint[index]    #in global frame
                 #print("point find ", index)
        return None
        
    def get_actuation(self, pose_theta, lookahead_point, position, lookahead_distance, wheelbase):
        # waypoint_car = np.dot(get_rotation_matrix(-pose_theta), (lookahead_point[0:2]-position))
        # waypoint_y = waypoint_car[1]
        waypoint_y = np.dot(np.array([np.sin(-pose_theta), np.cos(-pose_theta)]), lookahead_point[0:2]-position)
        
        if np.abs(waypoint_y) < 1e-6:
            return speed, 0.
        radius = 1/(2.0*waypoint_y/lookahead_distance**2)
        steering_angle = np.arctan(wheelbase/radius)
        speed = self.select_velocity(steering_angle)
        return speed, steering_angle


    def select_velocity(self, angle):
        if abs(angle) <= 5*math.pi/180:
            velocity  = 4
        elif abs(angle) <= 10*math.pi/180:
            velocity  = 4.0
        elif abs(angle) <= 15*math.pi/180:
            velocity = 4.0
        elif abs(angle) <= 20*math.pi/180:
            velocity = 4.0
        else:
            velocity = 3.0
        return velocity



    def plan(self, obs, action):
        #Choose the path to follow
        path = self.waypoints[action]      
        
        pose_x = obs['poses_x'][1]
        pose_y = obs['poses_y'][1]
        pose_theta = obs['poses_theta'][1]
        position = np.array([pose_x, pose_y])
        lookahead_point = self._get_current_waypoint(path, self.lookahead_distance, position, pose_theta)
        if lookahead_point is None:
            return self.safe_speed, 0.0
        speed, steering_angle = self.get_actuation(pose_theta, lookahead_point, position, self.lookahead_distance, self.wheelbase)
        return speed, steering_angle