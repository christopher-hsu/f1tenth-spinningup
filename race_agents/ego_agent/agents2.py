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
        self.waypoints = np.zeros((len(csv_path),1000,2))
        self.path_waypoints = np.zeros((len(csv_path),4))
        self.aval_paths = set(range(len(csv_path)))
        count = 0
        for path in csv_path:
            self.waypoints[count,:,:] = np.loadtxt(path, ndmin=2,delimiter=',')
            count += 1

 
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

        num_subsample = 117

        obs_array = np.zeros((256,))

        obs_array[239:] = 10 ## If lane is not accessible, this is the default distance to it

        ranges = np.array(list(obs['scans'][0]))
        angles = np.linspace(-4.7/2., 4.7/2., num=ranges.shape[0])

        min_idx = int(((-100)*(np.pi/180)-angles[0])/(angles[1]-angles[0]))
        max_idx = int(((100)*(np.pi/180)-angles[0])/(angles[1]-angles[0]))

        # lidar_idxs = np.random.randint(min_idx,max_idx,num_subsample)
        lidar_idxs = np.linspace(min_idx,max_idx,num=num_subsample).astype(int) #subsample lidar

        lidar_xy_our_frame = np.zeros((num_subsample,2))

        lidar_xy_our_frame[:,0] = (ranges[lidar_idxs] * np.cos(angles[lidar_idxs]))
        lidar_xy_our_frame[:,1] = (ranges[lidar_idxs] * np.sin(angles[lidar_idxs]))

        lidar_our_frame = lidar_xy_our_frame.flatten()  ## Sampled lidar readings shape (num_subsample*2,)
        obs_array[:num_subsample*2] = lidar_our_frame
        ## lets get other cars orientation with respect us

        ## opp position global to our frame
        our_position = np.array([obs['poses_x'][0],obs['poses_y'][0]])
        opp_car_global = np.array([obs['poses_x'][1],obs['poses_y'][1]])

        opp_car_global = opp_car_global - our_position

        R_mat = np.array([[np.cos(obs['poses_theta'][0]),np.sin(obs['poses_theta'][0])],
            [-np.sin(obs['poses_theta'][0]),np.cos(obs['poses_theta'][0])]])

        pos_opp_our_frame  = np.dot(opp_car_global,R_mat.T)  ## shape (1,2) that gives their position wrt us
        theta_opp_our_frame = obs['poses_theta'][0] - obs['poses_theta'][1]  # one value that gives their theta wrt us
        obs_array[num_subsample*2:(num_subsample*2)+2] = pos_opp_our_frame
        obs_array[(num_subsample*2)+2:(num_subsample*2)+3] = theta_opp_our_frame

        ## Now their velocity with respect to us

        # first their velocity to global frame:
        vel_opp_frame = np.array([obs['linear_vels_x'][1],0])
        R_mat_opp = np.array([[np.cos(obs['poses_theta'][1]),np.sin(obs['poses_theta'][1])],
            [-np.sin(obs['poses_theta'][1]),np.cos(obs['poses_theta'][1])]])

        vel_opp_global  = np.dot(vel_opp_frame,R_mat_opp)

        # Opp velocity global to our local frame:

        vel_opp_our_frame  = np.dot(vel_opp_global,R_mat.T) ## shape (1,2) that gives their velocity wrt us

        obs_array[(num_subsample*2)+3:(num_subsample*2)+5] = vel_opp_our_frame


        ## Now we gotta find our distance to each lane including optimal lane

        self.find_waypoints(our_position, obs['poses_theta'][0]) ## finds the waypoints from all paths and also find available paths

        for path in self.aval_paths:
            point1 = self.path_waypoints[path,:2]
            point2 = self.path_waypoints[path,2:]

            denom = np.sqrt((point2[1]-point1[1])**2 + (point2[0]-point1[0])**2)
            if denom == 0:
                continue
            num = np.abs(((point2[1]-point1[1])*our_position[0]) - ((point2[0]-point1[0])*our_position[1]) + (point2[0]*point1[1]) - (point2[1]*point1[0]))

            obs_array[((num_subsample*2)+5) + path] = num/denom

        return obs_array

        
    def get_actuation(self, pose_theta, lookahead_point, position):
        # waypoint_car = np.dot(get_rotation_matrix(-pose_theta), (lookahead_point[0:2]-position))
        # waypoint_y = waypoint_car[1]
        waypoint_y = np.dot(np.array([np.sin(-pose_theta), np.cos(-pose_theta)]), lookahead_point[0:2]-position)
        
        if np.abs(waypoint_y) < 1e-6:
            return self.safe_speed, 0.
        radius = 1/(2.0*waypoint_y/self.lookahead_distance**2)
        steering_angle = np.arctan(self.wheelbase/radius)
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
        # path = self.waypoints[action]      
        
        pose_x = obs['poses_x'][0]
        pose_y = obs['poses_y'][0]
        pose_theta = obs['poses_theta'][0]
        position = np.array([pose_x, pose_y])
        
        if action in self.aval_paths:
            lookahead_point = self.path_waypoints[action,:2]
            speed, steering_angle = self.get_actuation(pose_theta, lookahead_point, position)
        else:
            # raise Exception('Action is not accessible from here!')
            return 0.0, 0.0

        # lookahead_point = self._get_current_waypoint(path, self.lookahead_distance, position, pose_theta)
        # if lookahead_point is None:
        #     return self.safe_speed, 0.0
        # speed, steering_angle = self.get_actuation(pose_theta, lookahead_point, position)
        return speed, steering_angle