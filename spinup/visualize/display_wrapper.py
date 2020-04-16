from gym import Wrapper
import numpy as np
from numpy import linalg as LA

import pdb, os
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from matplotlib import patches
from matplotlib import animation

class Display2D(Wrapper):
    def __init__(self, env):
        super(Display2D, self).__init__(env)
        self.figID = 0
        self.fig = plt.figure(self.figID)
        self.map = np.zeros((50,50))

    def close(self):
        plt.close(self.fig)

    def render(self):
        obs = self.obs

        #plot stuff here like ax.plot
        self.fig.clf()  
        ax = self.fig.subplots()
        im = ax.imshow(self.map, cmap='gray_r', origin='lower',
                    extent=[-5, 5,-5,5])

        ax.plot(obs['poses_x'][0],obs['poses_y'][0], 'r.', markersize=2)


        plt.draw()
        plt.pause(0.0005)

        pass

    def reset(self, poses=None):

        return self.env.reset(poses)
