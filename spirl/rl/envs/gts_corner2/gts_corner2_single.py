from gym_gts import GTSApi
import gym

import numpy as np
import math

from spirl.rl.components.environment import GymEnv
from spirl.utils.general_utils import ParamDict, AttrDict

from spirl.utils.gts_utils import make_env, initialize_gts
from spirl.utils.gts_utils import  corner2_done_function, corner2_spare_reward_function

from spirl.utils.gts_utils import CAR_CODE, COURSE_CODE, TIRE_TYPE, BOP, DEFAULT_FEATURE_KEYS
from spirl.utils.gts_utils import start_condition_formulator


class GTSEnv_Corner2_Single(GymEnv):

    VIS_RANGE = [-10, 10]
    START_POS = [-420, -109]
    TARGET_POS = [416, 499]

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
            'num_cars' : 1,
            'disable_env_checker': True,
        })
        return super()._default_hparams().overwrite(default_dict)

    def _game_hp(self):
        game_hp = ParamDict({
            'do_init' : True,
            'reward_function' : corner2_spare_reward_function,
            'done_function' : corner2_done_function,
            'standardize_observations' : True,
            'store_states' : False,
            'builtin_controlled': [],
            'min_frames_per_action': 6,
            'initial_velocity': 200,
        })
        return game_hp


    def _initialize(self):
        bops = self._generate_bop()

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
            builtin_controlled = self._hp.builtin_controlled,
            store_states = self._hp.store_states,
            standardize_observations=self._hp.standardize_observations,
            
            # disable_env_checker =  self._hp.disable_env_checker,
        )

        self.course_length = self._get_course_length()

    def reset(self, start_conditions=None):
        self._reset_storage()
    
        if not start_conditions:
            start_conditions = self._formulate_start_conditions()
        obs = self._env.reset(start_conditions=start_conditions)
        return obs

    def step(self, actions):
        obs, rew, done, info = self._env.step(actions)
        self._post_step_by_info(info)
        return obs, rew, done, info

    def _get_course_length(self):
        course_length, course_code, course_name = self._env.get_course_meta()
        return course_length

    def render(self, mode='rgb_array'):
        return [[[[0,0,0]]] for _ in range(self._hp.num_cars)]

#  ========================= class methods ========================
    def _generate_bop(self):
        bops = [BOP[self._hp.car_name]] * self._hp.num_cars
        return bops

    def _formulate_start_conditions(self):
        course_v = [1200]
        speed = [self._hp.initial_velocity] * self._hp.num_cars
        start_conditions = start_condition_formulator(num_cars=self._hp.num_cars, course_v=course_v, speed=speed)
        return start_conditions

    def _reset_storage(self):
        self._lap_time = 0
        self._hit_wall_time = 0
        self._hit_car_time = 0

    def _post_step_by_info(self, info):
        self._lap_time = info[0]['state']['current_lap_time_msec'] / 1000
        self._hit_wall_time = info[0]['state']['hit_wall_time']
        self._hit_car_time = info[0]['state']['hit_cars_time']
        
        # if info[0]['state']['is_hit_wall']:
        #     print('hit wall at ', self._lap_time)
            
        # if info[0]['state']['is_hit_cars']:
        #     print('hit car at ', self._lap_time)

    def get_episode_info(self):
        return AttrDict(lap_time = self._lap_time,
                        hit_wall_time = self._hit_wall_time,
                        hit_car_time = self._hit_car_time)

