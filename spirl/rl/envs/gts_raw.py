from gym_gts import GTSApi
import gym

import numpy as np
import math

from spirl.rl.components.environment import GymEnv
from spirl.utils.general_utils import ParamDict, AttrDict

from spirl.utils.gts_utils import make_env, initialize_gts
from spirl.utils.gts_utils import reward_function, sampling_done_function

from spirl.utils.gts_utils import CAR_CODE, COURSE_CODE, TIRE_TYPE, BOP, DEFAULT_FEATURE_KEYS
from spirl.utils.gts_utils import start_condition_formulator


class GTSEnv_Raw(GymEnv):
    def __init__(self, config):
        self._hp = self._default_hparams()
        self._hp.overwrite(self._game_hp())
        self._hp.overwrite(config)

        if (self._hp.do_init):
            self._initialize()
        self._make_env()

    def _default_hparams(self):
        default_dict = ParamDict({
            'ip_address' : '192.168.1.5',
            'car_name' : 'Audi TTCup',
            'course_name' : 'Tokyo Central Outer' ,
            'num_cars' : 20,
        })
        return super()._default_hparams().overwrite(default_dict)

    def _game_hp(self):
        game_hp = ParamDict({
            'do_init' : True,
            'reward_function' : reward_function,
            'done_function' : sampling_done_function,
            'standardize_observations' : True,
            'min_frames_per_action': 6,
            'initial_velocity': 144,
        })
        return game_hp


    def _initialize(self):
        bops = [BOP[self._hp.car_name]] * self._hp.num_cars

        initialize_gts(ip = self._hp.ip_address,
                      num_cars=self._hp.num_cars, 
                      car_codes = CAR_CODE[self._hp.car_name], 
                      course_code = COURSE_CODE[self._hp.course_name], 
                      tire_type = TIRE_TYPE, 
                      bops = bops
                      )
    
    def _make_env(self):
        self._env = make_env(
            ip = self._hp.ip_address, 
            feature_keys = DEFAULT_FEATURE_KEYS,
            min_frames_per_action = self._hp.min_frames_per_action, 
            reward_function = self._hp.reward_function,
            done_function = self._hp.done_function,
            standardize_observations=self._hp.standardize_observations,
        )

        self.course_length = self._get_course_length()

    def reset(self, start_conditions=None):
    
        if not start_conditions:
            course_gap = math.floor(self.course_length / self._hp.num_cars)
            course_init = np.random.rand() * self.course_length
            course_v = [(course_init + course_gap*i)% self.course_length  for i in range(self._hp.num_cars)]
            speed = [self._hp.initial_velocity] * self._hp.num_cars
            start_conditions = start_condition_formulator(num_cars=self._hp.num_cars, course_v=course_v, speed=speed)
        obs = self._env.reset(start_conditions=start_conditions)
        return obs

    def step(self, actions):
        obs, rew, done, info = self._env.step(actions)
        return obs, rew, done, info

    def _get_course_length(self):
        course_length, course_code, course_name = self._env.get_course_meta()
        return course_length

    def render(self, mode='rgb_array'):
        return [[[[0,0,0]]] for _ in range(self._hp.num_cars)]