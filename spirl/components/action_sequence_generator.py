import numpy as np
import matplotlib.pyplot as plt

from spirl.utils.general_utils import AttrDict

class ActionSeqGen:
    def __init__(self, n_dim, n_step):
        self.n_dim = n_dim
        self.n_step = n_step
        self._hp = self.get_hp()

    def get_hp(self):
        hp = AttrDict(
            action_range = 1.0,
            number_of_slope_change = 1,
        )
        return hp

    def generate_action_seq(self):
        # generate the action sequence for the whole action space
        actions = []
        for _ in range(self.n_dim):
            actions.append(self.generate_seq())
        return np.float32(actions).reshape(self.n_step, self.n_dim)

    def generate_seq(self):
        # generate one sequence 
        # 1. generate initial value in the action range
        initial_value = self.generate_initial_value()
        initial_value = self.clip_range(initial_value)

        # 2. choose position where the slop change
        assert(self._hp.number_of_slope_change < self.n_step-1)
        change_slope_position = np.random.choice(np.arange(1,self.n_step), self._hp.number_of_slope_change, replace=False)

        # 3. generate changing value at each position
        slope = 1.0 if np.random.rand() > 0.5 else -1.0
        action_seq = [initial_value]

        for idx in range(self.n_step-1):
            slope = -slope if idx in change_slope_position else slope
            new_action = self.clip_range( action_seq[-1] + slope * self.generate_change_value() )
            action_seq.append(new_action)

        return np.float32(action_seq)

    def generate_initial_value(self):
        return np.random.randn() / 3 * self._hp.action_range

    def generate_change_value(self):
        return abs(np.random.randn() / 3 * self._hp.action_range) 

    def clip_range(self, value):
        return np.clip(value, -self._hp.action_range, self._hp.action_range)