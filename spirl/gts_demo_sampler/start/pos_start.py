from spirl.utils.general_utils import AttrDict, ParamDict

from spirl.gts_demo_sampler.start.base_start import BaseStart
from spirl.gts_demo_sampler.file.file_operation import load_file

import numpy as np
import pandas as pd

'''
1. Track centerline and save centerline path info
2. In the Frenet coordinate, sample s, ey, epsi
3. Convert to the Cartesian coordinate
4. Convert back to the value inversely (forward in gym_gts)
'''

''''
Now we can reuse the results from GTS-Master.
We can try to load it. It is a track class, we can also copy the source code here.
Then test loading it here.
'''

class PosStart(BaseStart):
    def __init__(self, config):
        self._hp = self._default_hparams().overwrite(config)

    def _default_hparams(self):
        default_dict = ParamDict({
            'num_cars' :        1,
            'pos' :             [[0,0,0]],
            'rot' :             [[0,0,0]],
            'speed_kmph':       [144],
        })
        return super()._default_hparams().overwrite(default_dict)

    def _generate_condition(self, idx):
        condition = {
            'id': idx,
            'pos': self._hp.pos[idx],
            'rot': self._hp.rot[idx],
            'speed_kmph': self._hp.speed_kmph[idx],
        }
        return condition



class PosStart_by_course_v(PosStart):
    def __init__(self, config):
        super().__init__(config)
        self._load_track()

    def _default_hparams(self):
        default_dict = ParamDict({
            'track_dir':        '',
            'course_v_range':           [0, 100],
            'speed_kmph_range':         [0, 144],
            'ey_range_percent':         [-0.5, 0.5], # half width
            'epsi_range_pi_percent':    [-0.5, 0.5] # +- pi/2, positive direction
        })
        return super()._default_hparams().overwrite(default_dict)

    def _generate_condition(self, idx):
        states = self._sample_states()
        # print('start states', states)
        state = self._inverse_gym_gts(states)
        condition = {
            'id': idx,
            'pos': state["pos"],
            'rot': state["rot"],
            'speed_kmph': states.speed_kmph,
        }
        return condition

    def _load_track(self):
        # self.track = load_file(self._hp.track_dir)
        data = pd.read_csv(self._hp.track_dir)
        data2 = {}
        for d in data:
            data2[d] = data[d].values
        self.track = AttrDict(data2)

    def _sample_states(self):
        course_v = np.random.random() * (self._hp.course_v_range[1] - self._hp.course_v_range[0]) + self._hp.course_v_range[0]
        speed_kmph = np.random.random() * (self._hp.speed_kmph_range[1] - self._hp.speed_kmph_range[0]) + self._hp.speed_kmph_range[0]
        X_cent = np.interp(course_v, self.track.s, self.track.X)
        Y_cent = np.interp(course_v, self.track.s, self.track.Y)
        Wl = np.interp(course_v, self.track.s, self.track.Wl)
        Wr = np.interp(course_v, self.track.s, self.track.Wr)
        Psi_cent = np.interp(course_v, self.track.s, self.track.Psi)
        Theta = np.interp(course_v, self.track.s, self.track.Theta)

        ey_percent = np.random.random() * (self._hp.ey_range_percent[1] - self._hp.ey_range_percent[0]) + self._hp.ey_range_percent[0]
        epsi_percent = np.random.random() * (self._hp.epsi_range_pi_percent[1] - self._hp.epsi_range_pi_percent[0]) + self._hp.epsi_range_pi_percent[0]

        if ey_percent > 0:
            ey = ey_percent * Wl
        else:
            ey = ey_percent * Wr
        epsi = epsi_percent * np.pi

        Frenet_states = AttrDict(
            course_v = course_v, 
            speed_kmph = speed_kmph,
            X_cent = X_cent,
            Y_cent = Y_cent,
            Psi_cent = Psi_cent,
            Theta = Theta,
            ey = ey,
            epsi = epsi,
        )
        self._convert_to_Cartesian(Frenet_states)
        return Frenet_states

    def _convert_to_Cartesian(self, states):
        states.X, states.Y = bias(states.X_cent , states.Y_cent , states.Psi_cent , states.ey )    
        
    def _inverse_gym_gts(self, states):
        Psi = states.Psi_cent + states.epsi
        rot1 = WrapToPi(np.pi/2 - Psi)
        state = {
            "pos" : [states.X, 0, -states.Y], # X, Z, Y
            "rot" : [-states.Theta, -rot1, 0], # Theta, Psi, Phi 
        }
        return state


    

    

def bias(X,Y,psi,W):  # prject to normal direction
    X2 = X - W * np.sin(psi)
    Y2 = Y + W * np.cos(psi)
    return X2,Y2

def WrapToPi(x):
    x = np.mod(x, 2*np.pi)
    x -= 2*np.pi * np.heaviside((x - np.pi), 0)
    return x