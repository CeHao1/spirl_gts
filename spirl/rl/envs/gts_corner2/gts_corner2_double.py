from gym_gts import GTSApi
import gym

import numpy as np
import copy

# from spirl.rl.components.environment import GymEnv
from spirl.rl.envs.gts_corner2.gts_corner2_single import GTSEnv_Corner2_Single
from spirl.utils.general_utils import ParamDict, AttrDict

from spirl.utils.gts_utils import make_env, initialize_gts
from spirl.utils.gts_utils import double_reward_function, corner2_done_function, judge_overtake_success_info

from spirl.utils.gts_utils import CAR_CODE, COURSE_CODE, TIRE_TYPE, BOP, DEFAULT_FEATURE_KEYS
from spirl.utils.gts_utils import start_condition_formulator


# consider a special value of successful overtake

class GTSEnv_Corner2_Double(GTSEnv_Corner2_Single):
    def __init__(self, config):
        self._hp = self._default_hparams()
        self._hp.overwrite(self._game_hp())
        self._hp.overwrite(config)

        if (self._hp.do_init):
            self._initialize()
        self._make_env()

    def _default_hparams(self):
        default_dict = ParamDict({
            'ip_address' : '192.168.1.100',
            'car_name' : 'Audi TTCup',
            'course_name' : 'Tokyo Central Outer' ,
            'num_cars' : 2,
            'disable_env_checker': True,
        })
        return super()._default_hparams().overwrite(default_dict)

    def _game_hp(self):
        game_hp = ParamDict({
            'do_init' : True,
            'reward_function' : double_reward_function,
            'done_function' : corner2_done_function,
            'standardize_observations' : False,
            'store_states' : False,
            'builtin_controlled': [0],
            'min_frames_per_action': 6,
            'initial_velocity': [200, 144],
            'initial_course_v': [1200, 1400], 
            'feature_keys': DEFAULT_FEATURE_KEYS,
            'bop': [[1, 1], [0.8, 1.2]],
        })
        return game_hp

    
    def _make_env(self):
        self._env = make_env(
            ip = self._hp.ip_address, 
            feature_keys = self._hp.feature_keys,
            min_frames_per_action = self._hp.min_frames_per_action, 
            reward_function = self._hp.reward_function,
            done_function = self._hp.done_function,
            builtin_controlled = self._hp.builtin_controlled,
            store_states = self._hp.store_states,
            standardize_observations=self._hp.standardize_observations,
            
            disable_env_checker =  self._hp.disable_env_checker,
        )

        self.course_length = self._get_course_length()

    def _generate_bop(self):
        if self._hp.bop is None:
            self.bops = [BOP[self._hp.car_name]] * self._hp.num_cars
        else:
            assert len(self._hp.bop) == self._hp.num_cars
            bop_template = BOP[self._hp.car_name]
            self.bops = []

            for idx in range(self._hp.num_cars):
                bop = copy.deepcopy(bop_template)
                bop['power'] = round(bop['power'] * self._hp.bop[idx][0])
                bop['weight'] = round(bop['weight'] * self._hp.bop[idx][1])
                self.bops.append(bop)

    def _formulate_start_conditions(self):
        course_v = self._hp.initial_course_v
        speed = self._hp.initial_velocity
        start_conditions = start_condition_formulator(num_cars=self._hp.num_cars, course_v=course_v, speed=speed)
        return start_conditions

    def _reset_storage(self):
        super()._reset_storage()
        self._success = 0
        self._success_course = 0

    def _post_step_by_info(self, info):
        if not self._success and judge_overtake_success_info(info):
            self._success = 1
            self._success_course = info[0]['state']['course_v']
            print('successful overtake')

    def get_episode_info(self):
        return super().get_episode_info().update(
            AttrDict(success = self._success,
                    success_course = self._success_course))