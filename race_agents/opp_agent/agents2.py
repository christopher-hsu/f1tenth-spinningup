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
        self.lookahead_distance = 1
        self.safety_radius = 0.15
        self.TTC_threshold= 0.65
        self.minTTCspeed = 3.0
        self.wheelbase = wheelbase
        self.waypoints = np.zeros((len(csv_path),1000,2))
        self.path_waypoints = np.zeros((len(csv_path),4))
        self.aval_paths = set(range(len(csv_path)))
        count = 0
        for path in csv_path:
            self.waypoints[count,:,:] = np.loadtxt(path, ndmin=2,delimiter=',')
            count += 1

    def find_waypoints(self,current_position, current_theta):

        point_dist =  np.sqrt(np.sum(np.square(self.waypoints[:, :,0:2]-current_position), axis=2))
        
        point_index = np.where(np.abs(point_dist-self.lookahead_distance)< 0.2)

        car_points = self.global_to_car(self.waypoints[point_index], current_position,current_theta)

        abs_tan_vals = np.abs(np.arctan(car_points[:,0]/car_points[:,1]))

        good_points = np.all([abs_tan_vals < np.pi/2, car_points[:,0]>0],axis = 0)

        waypoint_idx_paths = {k: v for v, k in enumerate(point_index[0][good_points])}
        self.test = point_index[0][good_points]
        for key in waypoint_idx_paths.keys():
            self.path_waypoints[key,:2] = self.waypoints[key,point_index[1][good_points][waypoint_idx_paths[key]],:]
            self.path_waypoints[key,2:] = self.waypoints[key,point_index[1][good_points][waypoint_idx_paths[key]-1],:]

        self.aval_paths = set(point_index[0][good_points])


    @staticmethod
    def global_to_car(points, current_position, current_theta):

        points = points - current_position

        R_mat = np.array([[np.cos(current_theta),np.sin(current_theta)],
                        [-np.sin(current_theta),np.cos(current_theta)]])

        car_points = np.dot(points,R_mat.T)


        return car_points

    def process_obs(self,obs):

        '''
        Args: 
            obs: observation that we get out of each step
        Returns:
            obs_array: (numpy array shape (256,)) I also find waypoints and available paths here
        '''

        ## First lets subsample from the lidar: lets do

        num_subsample = 234

        obs_array = np.zeros((256,))

        obs_array[239:] = 10 ## If lane is not accessible, this is the default distance to it
        """
            Main difference in this oppenent agent file
            Uses the second (opponent) car's lidar scans.
            And positions are the opponent poses.
            "our" frame is the opponent

        """
        self.speed = obs['linear_vels_x'][1]

        ranges = np.array(list(obs['scans'][1]))
        self.angles = np.linspace(-4.7/2., 4.7/2., num=ranges.shape[0])

        min_idx = int(((-100)*(np.pi/180)-self.angles[0])/(self.angles[1]-self.angles[0]))
        max_idx = int(((100)*(np.pi/180)-self.angles[0])/(self.angles[1]-self.angles[0]))

        lidar_idxs = np.linspace(min_idx,max_idx,num=num_subsample).astype(int) #subsample lidar
        obs_array[:num_subsample] = ranges[lidar_idxs]


        ## lets get other cars orientation with respect us
        ## opp position global to our frame
        ego_position = np.array([obs['poses_x'][0],obs['poses_y'][0]])
        opp_car_global = np.array([obs['poses_x'][1],obs['poses_y'][1]])

        # opp_car_global = opp_car_global - ego_position
        ego_car_global = ego_position - opp_car_global

        R_mat = np.array([[np.cos(obs['poses_theta'][1]),np.sin(obs['poses_theta'][1])],
            [-np.sin(obs['poses_theta'][1]),np.cos(obs['poses_theta'][1])]])

        pos_ego_our_frame  = np.dot(ego_car_global,R_mat.T)  ## shape (1,2) that gives their position wrt us
        theta_ego_our_frame = obs['poses_theta'][1] - obs['poses_theta'][0]   # one value that gives their theta wrt us
        obs_array[num_subsample:(num_subsample)+2] = pos_ego_our_frame
        obs_array[(num_subsample)+2:(num_subsample)+3] = theta_ego_our_frame

        ## Now their velocity with respect to us

        # first their velocity to global frame:
        vel_ego_frame = np.array([obs['linear_vels_x'][0],0])
        R_mat_ego = np.array([[np.cos(obs['poses_theta'][0]),np.sin(obs['poses_theta'][0])],
            [-np.sin(obs['poses_theta'][0]),np.cos(obs['poses_theta'][0])]])

        vel_ego_global  = np.dot(vel_ego_frame,R_mat_ego)

        # Ego velocity global to our local frame:

        vel_ego_our_frame  = np.dot(vel_ego_global,R_mat.T) ## shape (1,2) that gives their velocity wrt us

        obs_array[(num_subsample)+3:(num_subsample)+5] = vel_ego_our_frame


        ## Now we gotta find our distance to each lane including optimal lane

        self.find_waypoints(opp_car_global, obs['poses_theta'][1]) ## finds the waypoints from all paths and also find available paths

        for path in self.aval_paths:
            point1 = self.path_waypoints[path,:2]
            point2 = self.path_waypoints[path,2:]

            denom = np.sqrt((point2[1]-point1[1])**2 + (point2[0]-point1[0])**2)
            if denom == 0:
                continue
            num = np.abs(((point2[1]-point1[1])*ego_position[0]) - ((point2[0]-point1[0])*ego_position[1]) + (point2[0]*point1[1]) - (point2[1]*point1[0]))

            obs_array[((num_subsample)+5) + path] = num/denom

        return obs_array

        
    def find_TTC(self,waypoint):
        goalx_veh = waypoint[0]
        goaly_veh = waypoint[1]

        waypoint_angle_car = np.arctan(goaly_veh/goalx_veh)


        waypoint_angle_car_idx = int((waypoint_angle_car+np.pi)/(self.angles[1]-self.angles[0]))
        waypoint_angle_car_idx = np.minimum(waypoint_angle_car_idx,self.ranges.shape[0]-1)
        waypoint_angle_car_idx = np.maximum(waypoint_angle_car_idx,0)
        min_val = self.ranges[waypoint_angle_car_idx]

        angle_circle = np.arctan(self.safety_radius/min_val)

        if waypoint_angle_car<0:
            min_idx = int((waypoint_angle_car-angle_circle-self.angles[0])/(self.angles[1]-self.angles[0]))
            max_idx = int((waypoint_angle_car+angle_circle-self.angles[0])/(self.angles[1]-self.angles[0]))
            # print('min ', min_idx, ' ', max_idx)
        else:
            min_idx = int((waypoint_angle_car-angle_circle-self.angles[0])/(self.angles[1]-self.angles[0]))
            max_idx = int((waypoint_angle_car+angle_circle-self.angles[0])/(self.angles[1]-self.angles[0]))

        lower_idx_0 = np.maximum(min_idx,0)    ##Look into this because of wrapping
        upper_idx_0 = np.minimum(max_idx,self.ranges.shape[0]-1)  ##Look into this because of wrapping

        TTC_denom = (np.maximum(self.minTTCspeed, self.speed)*np.cos(self.angles[lower_idx_0:upper_idx_0]))

        TTC_denom[TTC_denom<=0] = 0.00000001

        TTC_vals = self.ranges[lower_idx_0:upper_idx_0]/TTC_denom

        TTC_min = np.amin(TTC_vals)

        return TTC_min

    def get_actuation(self, pose_theta, lookahead_point, position, TTC=True):

        goal_veh= self.global_to_car(lookahead_point, position, pose_theta)

        current_TTC = self.find_TTC(goal_veh)

        newaction = -1
        if TTC == True:
            ##TTC avoidance
            if current_TTC <= self.TTC_threshold:
                Path_TTC_vals = {}
                goal_vals = {}
                for path_idx in self.aval_paths:
                    goal_vals[path_idx]= self.global_to_car(self.path_waypoints[path_idx,:2],position, pose_theta)
                    Path_TTC_vals[path_idx] = self.find_TTC(goal_vals[path_idx])

                newaction = max(Path_TTC_vals, key=Path_TTC_vals.get)
                # print('Chose: ', newaction)
                lookahead_point =self.path_waypoints[newaction,:2]
                goal_veh= goal_vals[newaction]

        L = np.sqrt((lookahead_point[0]-position[0])**2 +  (lookahead_point[1]-position[1])**2 )

        if np.abs(L) < 1e-6:
            return self.safe_speed, 0.
        arc = 2*goal_veh[1]/(L**2)
        angle = 0.33*arc
        steering_angle = np.clip(angle, -0.4, 0.4)
        speed = self.select_velocity(steering_angle)

        return speed, steering_angle, newaction

    def select_velocity(self, angle):
        if abs(angle) <= 5*math.pi/180:
            velocity  = 4.5
        elif abs(angle) <= 10*math.pi/180:
            velocity  = 4
        elif abs(angle) <= 15*math.pi/180:
            velocity = 3
        elif abs(angle) <= 20*math.pi/180:
            velocity = 2.5
        else:
            velocity = 2
        return velocity



    def plan(self, obs, action):
        ## Choose the path to follow        
        ## Opponent plan
        pose_x = obs['poses_x'][1]
        pose_y = obs['poses_y'][1]
        pose_theta = obs['poses_theta'][1]
        position = np.array([pose_x, pose_y])
        self.ranges = np.array(list(obs['scans'][1]))

        if action in self.aval_paths:
            lookahead_point = self.path_waypoints[action,:2]
            speed, steering_angle, newaction = self.get_actuation(pose_theta, lookahead_point, position)
            ## If action was adjusted with TTC
            if newaction != -1:
                action = newaction
        else:
            # raise Exception('Action is not accessible from here!')
            return 0.0, 0.0, action

        return speed, steering_angle, action