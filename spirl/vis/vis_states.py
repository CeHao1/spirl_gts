
import imp
from spirl.utils.general_utils import ParamDict, AttrDict
from spirl.utils.gts_utils import load_replay_2_states, load_track

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# we expect to visualize the results of built-in AI, human expert, and saved trajectory

'''
course from 12000 to 2400. time about 21-23s

'''
colors = {'human': 'k',
          'builtin': 'g--',
          'sac': 'b',
          'skill-critic': 'r',
          }

class VisStates:

    def __init__(self, config):
        self._hp = self._default_hparams().overwrite(config)

        self._load_track()
        self.load_replays()
        self.vis_corner2()

    def _default_hparams(self):
        # path of replays
        default_dict = ParamDict({
            'slim':                  [1200, 2400],
            'file_dir':             '',
            'track_dir':            '',
            'figure_path':          '',
            'human_file_name':      '',
            'builtin_file_name':    '',
            'sac_file_name':        '',
            'sc_file_name':         '',
        })
        return default_dict


    def load_replays(self):
        self.states = {}
        if self._hp.human_file_name:
            self.states['human'] = load_replay_2_states(self._hp.file_dir, self._hp.human_file_name, method='csv', chosen_lap=1)
            print('load human')
        
        if self._hp.builtin_file_name:
            self.states['builtin'] = load_replay_2_states(self._hp.file_dir, self._hp.builtin_file_name)
            print('load builtin')

        if self._hp.sac_file_name:
            self.states['sac'] = load_replay_2_states(self._hp.file_dir, self._hp.sac_file_name)
            print('load sac')

        if self._hp.sc_file_name:
            self.states['skill-critic'] = load_replay_2_states(self._hp.file_dir, self._hp.sc_file_name)
            print('load skill-critic')

    def _load_track(self):
        self.track = load_track(self._hp.track_dir)
        print('load track')


    def vis_corner2(self):
        self.calculate_time(slim=self._hp.slim)
        
        # to show path, ,
        self.show_map(slim=self._hp.slim)        

        # to show ey
        self.show_states(state_name='ey', slim=self._hp.slim)

        # to show Vx
        self.show_states(state_name='Vx', slim=self._hp.slim)

        # to show thr/brk and detla
        self.show_states(state_name='thr-brk', slim=self._hp.slim)
        self.show_states(state_name='delta', slim=self._hp.slim)
        

    def show_map(self, slim):

        track = self.track
       
        st, ed = slim[0], slim[1]
        idx_course = [True if s>st and s<ed else False for s in track.s]
        x_lim = [min(track.X[idx_course]), max(track.X[idx_course])]
        y_lim = [min(track.Y[idx_course]), max(track.Y[idx_course])]


        figsize = (10,10)

        plt.figure(figsize=figsize)
        fs = 20

        # centerline, boundary
        plt.plot(track.X, track.Y, 'g--', label='Track')
        plt.plot(track.Left_X, track.Left_Y, 'g', label='Boundary')
        plt.plot(track.Right_X, track.Right_Y, 'g')

        # path X, Y, 
        for name in self.states:
            plt.plot(self.states[name]['X'], self.states[name]['Y'], color=colors[name], label=name)

        # config
        plt.axis('equal')

        plt.legend(fontsize=fs)
        plt.xlim(x_lim)
        plt.ylim(y_lim)
        # plt.title('Overview of path on the track', fontsize=fs)
        plt.xlabel('X / m')
        plt.ylabel('Y / m')

        plt.savefig(self._hp.figure_path + 'map' + '.svg', format='svg',bbox_inches='tight')
        plt.show()
        
        
    def calculate_time(self, slim):
        for name in self.states:
            st, ed = slim[0], slim[1]
            idx_course = [True if s>st and s<ed else False for s in self.states[name]['s']]
            time = self.states[name]['t'][idx_course]
            d_t = time[-1] - time[0]
            print('time of {} is {}'.format(name, d_t))
            

    def show_states(self, state_name, slim):
        st, ed = slim[0], slim[1]
    
        plt.figure(figsize=(9,5))
        fs = 15

        for name in self.states:
            idx_course = [True if s>st and s<ed else False for s in self.states[name]['s']]
            plt.plot(self.states[name]['s'][idx_course], self.states[name][state_name][idx_course], color=colors[name], label=name)

        # plt.xlim(slim)
        plt.title(state_name, fontsize=fs)
        plt.legend(bbox_to_anchor=(0.0, 1.1), loc='lower left', fontsize=fs)
        plt.savefig(self._hp.figure_path + state_name + '.svg', format='svg',bbox_inches='tight')
        plt.show()
        
        
if __name__ == "__main__":
    config = ParamDict({
            'file_dir':             os.path.join(os.environ["EXP_DIR"], "skill_prior_learning/gts"),
            'track_dir':            os.path.join(os.environ["EXP_DIR"], "skill_prior_learning/gts/track.csv"),
            'figure_path':          os.path.join(os.environ["EXP_DIR"], "skill_prior_learning/gts/figures/"),
            'human_file_name':      'human.csv',
            'builtin_file_name':    '',
            'sac_file_name':        'sac.h5',
            'sc_file_name':         '',
        })
    vis = VisStates(config)
    