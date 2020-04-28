from gym import Wrapper
import numpy as np
from numpy import linalg as LA
from PIL import Image
import pdb, os, yaml
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from matplotlib import patches
from matplotlib import animation

class Display2D(Wrapper):
    def __init__(self, env, map_path, img_ext, rgb, flip):
        super(Display2D, self).__init__(env)
        self.figID = 0
        self.fig = plt.figure(self.figID)
        # self.map = np.zeros((50,50))

        self.init_map(map_path, img_ext, rgb, flip)

    def close(self):
        plt.close(self.fig)

    def render(self):
        obs = self.obs

        #plot stuff here like ax.plot
        self.fig.clf()  
        ax = self.fig.subplots()


        x_off = self.origin[0]*self.map_resolution
        y_off = self.origin[1]*self.map_resolution

        x_bound = (self.map_width)*self.map_resolution/2
        y_bound = (self.map_height)*self.map_resolution/2

        im = ax.imshow((1. - self.map_img/255.), cmap='gray_r', origin='lower',
                        extent=[-x_bound, x_bound+x_off, 
                                -y_bound, y_bound+y_off])#


        # grid = self.PosToMap(obs['poses_x'][0],obs['poses_y'][0])
        # grid_x, grid_y = self.se2_to_cell(obs['poses_x'][1],obs['poses_y'][1])
        # ax.plot(grid,grid_y, marker=(3, 0, obs['poses_theta'][0]/np.pi*180-90),
        #             markersize=10, linestyle='None', markerfacecolor='b', markeredgecolor='b')
        # ax.plot(obs['poses_x'][1],obs['poses_y'][1], 'r.', markersize=8)
        # ax.plot(obs['poses_x'][1],obs['poses_y'][1], marker=(3, 0, obs['poses_theta'][1]/np.pi*180-90),
                    # markersize=10, linestyle='None', markerfacecolor='r', markeredgecolor='r')

        # print(obs['poses_x'][0])
        # ax.plot(obs['poses_x'][0],obs['poses_y'][0], 'g.', markersize=8)
        ax.plot(obs['poses_x'][0],obs['poses_y'][0], marker=(3, 0, obs['poses_theta'][0]/np.pi*180-90),
                    markersize=10, linestyle='None', markerfacecolor='b', markeredgecolor='b')
        # ax.plot(obs['poses_x'][1],obs['poses_y'][1], 'r.', markersize=8)
        ax.plot(obs['poses_x'][1],obs['poses_y'][1], marker=(3, 0, obs['poses_theta'][1]/np.pi*180-90),
                    markersize=10, linestyle='None', markerfacecolor='r', markeredgecolor='r')

        # import pdb;pdb.set_trace()
        plt.draw()
        plt.pause(0.0005)

        pass

    def reset(self, poses=None):

        return self.env.reset(poses)


    def se2_to_cell(self, pos_x, pos_y):
        # pos = pos[:2]
        pos = np.array([pos_x,pos_y])
        cell_idx = (pos - self.mapmin)/self.mapres - 0.5
        return round(cell_idx[0]), round(cell_idx[1])
    
    def cell_to_se2(self, cell_idx):
        return ( np.array(cell_idx) + 0.5 ) * self.mapres + self.mapmin

    def PosToMap(self,poses_x, poses_y):
        """
            Takes numpy array of global positions and returns their indecies for
            our occupancy grid
        Args: 
            positions (numpy array shape (n*2)): global positions from ros
        Returns:
            grid_indecies (numpy array shape (2*n)): indicies of 
        """
        grid_indecies = np.zeros((2,2)).astype(int)
        # import pdb;pdb.set_trace()
        grid_indecies[1,:] = np.clip((int((poses_y - self.origin[0])/self.map_resolution)),0,self.map_height-1)
        grid_indecies[0,:] = np.clip(((int((poses_x - self.origin[1])/self.map_resolution))),0,self.map_width-1)
        return grid_indecies


    def init_map(self, map_path, img_ext, rgb, flip):
        """
            init a map for the gym env
            map_path: full path for the yaml, same as ROS, img and yaml in same dir
            rgb: map grayscale or rgb
            flip: if map needs flipping
        """

        self.map_path = map_path
        if not map_path.endswith('.yaml'):
            print('Gym env - Please use a yaml file for map initialization.')
            print('Exiting...')
            sys.exit()

        # split yaml ext name
        map_img_path = os.path.splitext(self.map_path)[0] + img_ext
        self.map_img = np.array(Image.open(map_img_path).transpose(Image.FLIP_TOP_BOTTOM))
        self.map_img = self.map_img.astype(np.float64)
        if flip:
            self.map_img = self.map_img[::-1]

        if rgb:
            self.map_img = np.dot(self.map_img[..., :3], [0.29, 0.57, 0.14])

        # update map metadata
        self.map_height = self.map_img.shape[0]
        self.map_width = self.map_img.shape[1]
        self.free_thresh = 0.6  # TODO: double check
        with open(self.map_path, 'r') as yaml_stream:
            try:
                map_metadata = yaml.safe_load(yaml_stream)
                self.map_resolution = map_metadata['resolution']
                self.origin = map_metadata['origin']
            except yaml.YAMLError as ex:
                print(ex)
        self.map_inited = True