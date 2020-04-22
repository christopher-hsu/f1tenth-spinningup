import os
import numpy as np
import math


class Agent(object):
    def __init__(self, csv_path):
        # TODO: load waypoints from csv
        self.waypoints = None
        self.safe_speed = 0.5
        
    def plan(self, obs):
        pass


class PurePursuit(Agent):
    def __init__(self, csv_path):
        self.index = 0
        self.FORWARD = 0.85
        self.waypoint = np.loadtxt(csv_path, ndmin=2,delimiter=',')
    

    def pose_callback(self, waypoints, position, theta):

        R_mat_ego = np.array([[np.cos(theta),np.sin(theta)],
                              [-np.sin(theta),np.cos(theta)]])

        point_dist =  np.sqrt(np.sum(np.square(waypoint[:, 0:2]-position), axis=1))
        point_index = np.where(abs(point_dist-FORWARD)< 0.20)[0]
        #print(point_index)
        for index in point_index:
            l2_0 = [waypoint[index, 0]-position[0], waypoint[index,1]-position[1]]
            goalx_veh = math.cos(euler[2])*l2_0[0] + math.sin(euler[2])*l2_0[1]
            goaly_veh = -math.sin(euler[2])*l2_0[0] + math.cos(euler[2])*l2_0[1]

            if abs(math.atan(goalx_veh/goaly_veh)) <  np.pi/2 and goalx_veh>0 :
                 self.waypoint = waypoint[index] 
                 #print("point find ", index)
                 break

        
        l2_0 = [self.waypoint[0]-position[0], self.waypoint[1]-position[1]]
        goalx_veh = math.cos(euler[2])*l2_0[0] + math.sin(euler[2])*l2_0[1]
        goaly_veh = -math.sin(euler[2])*l2_0[0] + math.cos(euler[2])*l2_0[1]  

        # TODO: calculate curvature/steering angle
        L = math.sqrt((self.waypoint[0]-position[0])**2 +  (self.waypoint[1]-position[1])**2 )

        arc = 2*goaly_veh/(L**2)
        angle = 0.3*arc
        angle = np.clip(angle, -0.35, 0.35)
        velocity = self.select_velocity(angle)

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

    def plan(self, action, obs):
        pose_x = obs['poses_x'][1]
        pose_y = obs['poses_y'][1]
        pose_theta = obs['poses_theta'][1]
        position = np.array([pose_x, pose_y])
        lookahead_point = self._get_current_waypoint(self.waypoints, self.lookahead_distance, position, pose_theta)
        if lookahead_point is None:
            return self.safe_speed, 0.0
        speed, steering_angle = get_actuation(pose_theta, lookahead_point, position, self.lookahead_distance, self.wheelbase)
        return speed, steering_angle