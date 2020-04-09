from gym import Wrapper
import numpy as np
from numpy import linalg as LA

import pdb, os
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from matplotlib import patches
from matplotlib import animation
from TTenv.TTcont.target_tracking.modules.metadata import *

class Display2D(Wrapper):
    def __init__(self, env, figID = 0, skip = 1, confidence=0.95):
        super(Display2D, self).__init__(env)
        self.figID = figID # figID = 0 : train, figID = 1 : test
        self.env_core = env.env
        self.bin = self.env_core.MAP.mapres
        if self.env_core.MAP.map is None:
            self.map = np.zeros(self.env_core.MAP.mapdim)
        else:
            self.map = self.env_core.MAP.map
        self.mapmin = self.env_core.MAP.mapmin
        self.mapmax = self.env_core.MAP.mapmax
        self.mapres = self.env_core.MAP.mapres
        #self.uncertaintyMap = np.log(30.0)*np.ones((self.mapdim, self.mapdim))
        self.fig = plt.figure(self.figID)
        self.n_frames = 0 
        self.skip = skip
        self.c_cf = np.sqrt(-2*np.log(1-confidence))
           
    def pos2map(self, obs, sd):
        x = obs[:,0]
        y = obs[:,1]
        # indx = np.rint((x + METADATA['limit'][0])*self.bin) 
        # indy = np.rint((y + METADATA['limit'][1])*self.bin) 
        # self.uncertaintyMap[indy.astype(np.int32), indx.astype(np.int32)] = np.maximum(np.log(sd), -100)

    def close(self):
        plt.close(self.fig)

    def render(self, mode='empty', record=False, traj_num=0, batch_outputs=None):
        state = self.env_core.agent.state
        belief = False
        if hasattr(self, 'belief_targets'):
            belief = True
        #if batch_outputs is not None:
        #    self.pos2map(obs = batch_outputs[0], sd = batch_outputs[1])  
        num_targets = len(self.traj_y)
        if type(self.env_core.targets) == list:
            target_true_pos = [self.env_core.targets[i].state[:2] for i in range(num_targets)]
            if belief == True:
                target_b_state = [self.env_core.belief_targets[i].state for i in range(num_targets)] # state[3:5]
                target_cov = [self.env_core.belief_targets[i].cov for i in range(num_targets)]
        else:
            target_true_pos = self.env_core.targets.state[:,:2]
            if belief == True:
                target_b_state = self.env_core.belief_targets.state[:,:2]  # state[3:5]
                target_cov = self.env_core.belief_targets.cov 
            
        if self.n_frames%self.skip == 0:
            self.fig.clf()       
            ax = self.fig.subplots()
            im = None
             # if mode == 'uncertainty':
            #     im = ax.imshow(self.uncertaintyMap, interpolation='gaussian',
            #         extent=(-METADATA['limit'][0]-0.5*self.bin, METADATA['limit'][0]+0.5*self.bin, 
            #             -METADATA['limit'][1]-0.5*self.bin, METADATA['limit'][1]+0.5*self.bin), 
            #         origin='lower', alpha=0.5, cmap = 'magma')
            #     im.set_clim(-7., np.log(30.0))
            #     plt.colorbar(mappable=im)

            if mode == 'empty':
                im = ax.imshow(self.map, cmap='gray_r', origin='lower',
                    extent=[self.mapmin[0], self.mapmax[0], self.mapmin[1], self.mapmax[1]])

            ax.plot(state[0], state[1], marker=(3, 0, state[2]/np.pi*180-90), markersize=10, 
                linestyle='None', markerfacecolor='b', markeredgecolor='b')
            ax.plot(self.traj[0], self.traj[1], 'b.', markersize=2)

            for i in range(num_targets):
                ax.plot(self.traj_y[i][0], self.traj_y[i][1], 'r.', markersize=2)
                ax.plot(target_true_pos[i][0], target_true_pos[i][1], marker='o', markersize=5, 
                    linestyle='None', markerfacecolor='r', markeredgecolor='r')
                # Belief on target
                if belief == True:
                    ax.plot(target_b_state[i][0], target_b_state[i][1], marker='o', markersize=10, 
                        linewidth=5, markerfacecolor='none', markeredgecolor='g')
                    # eig_val, eig_vec = LA.eig(target_cov[i])
                    eig_val, eig_vec = LA.eig(target_cov[i][:2,:2])
                    belief_target = patches.Ellipse((target_b_state[i][0], target_b_state[i][1]), 
                                2*np.sqrt(eig_val[0])*self.c_cf, 2*np.sqrt(eig_val[1])*self.c_cf, 
                                angle = 180/np.pi*np.arctan2(eig_vec[0][1],eig_vec[0][0]) ,fill=True,
                                zorder=2, facecolor='g', alpha=0.5)
                    ax.add_patch(belief_target)

            sensor_arc = patches.Arc((state[0], state[1]), METADATA['sensor_r']*2, METADATA['sensor_r']*2, 
                angle = state[2]/np.pi*180, theta1 = -METADATA['fov']/2, theta2 = METADATA['fov']/2, facecolor='gray')
            ax.add_patch(sensor_arc)
            ax.plot([state[0], state[0]+METADATA['sensor_r']*np.cos(state[2]+0.5*METADATA['fov']/180.0*np.pi)],
                [state[1], state[1]+METADATA['sensor_r']*np.sin(state[2]+0.5*METADATA['fov']/180.0*np.pi)],'k', linewidth=0.5)
            ax.plot([state[0], state[0]+METADATA['sensor_r']*np.cos(state[2]-0.5*METADATA['fov']/180.0*np.pi)],
                [state[1], state[1]+METADATA['sensor_r']*np.sin(state[2]-0.5*METADATA['fov']/180.0*np.pi)],'k', linewidth=0.5)
            
            ax.set_xlim((self.mapmin[0], self.mapmax[0]))
            ax.set_ylim((self.mapmin[1], self.mapmax[1]))
            ax.set_aspect('equal','box')
            ax.grid()
            ax.set_title(' '.join([mode.upper(),': Trajectory',str(traj_num)]))

            # fig0, ax0 = plt.subplots(num=2)
            # im_size = self.env_core.im_size
            # local_mapmin = np.array([-im_size/2*self.mapres[0], 0.0])
            # im0 = ax0.imshow(np.reshape(self.env_core.local_map, (im_size,im_size)), 
            #             cmap='gray_r', origin='lower', 
            #             extent=[local_mapmin[0], -local_mapmin[0], 0.0, -local_mapmin[0]*2])
            if not record :
                plt.draw()
                plt.pause(0.0005)
        self.n_frames += 1 
        self.traj[0].append(state[0])
        self.traj[1].append(state[1])
        for i in range(num_targets):
            self.traj_y[i][0].append(target_true_pos[i][0])
            self.traj_y[i][1].append(target_true_pos[i][1])

    def reset(self, **kwargs):
        self.traj = [[],[]]
        self.traj_y = [[[],[]]]*self.env_core.num_targets
        return self.env.reset(**kwargs)

class Video2D(Wrapper):
    def __init__(self, env, dirname = '', skip = 1, dpi=80):
        super(Video2D, self).__init__(env)
        self.skip = skip
        self.moviewriter = animation.FFMpegWriter()
        fname = os.path.join(dirname, 'train_%d.mp4'%np.random.randint(0,20))
        self.moviewriter.setup(fig=env.fig, outfile=fname, dpi=dpi)
        self.n_frames = 0

    def render(self, traj_num=0, *args, **kwargs):
        if self.n_frames % self.skip == 0:
        #if traj_num % self.skip == 0:
            self.env.render(record=True, traj_num=traj_num, *args, **kwargs)
        self.moviewriter.grab_frame()
        self.n_frames += 1

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

