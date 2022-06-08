from gym_gts import GTSApi
import gym

import numpy as np

from spirl.rl.components.environment import GymEnv
from spirl.utils.general_utils import ParamDict, AttrDict

from spirl.utils.gts_utils import make_env, initialize_gts
from spirl.utils.gts_utils import RL_OBS_1, CAR_CODE, COURSE_CODE, TIRE_TYPE, BOP
from spirl.utils.gts_utils import DEFAULT_FEATURE_KEYS
from spirl.utils.gts_utils import start_condition_formulator
from spirl.utils.gts_utils import raw_observation_to_true_observation

from spirl.utils.gts_utils import reward_function, sampling_done_function
from spirl.utils.gts_utils import eval_time_trial_done_function, eval_time_trial_reward_function

class GTSEnv_Base(GymEnv):
    def __init__(self, config):
        self._hp = self._default_hparams()
        self._hp.overwrite(self._game_hp())
        self._hp.overwrite(config)

        if (self._hp.do_init):
            self._initialize()
        self._make_env()

        self.state_scaler = None
        self.action_scaler = None
        self._load_standard_table()

    
    def _default_hparams(self):
        default_dict = ParamDict({
            'ip_address' : '192.168.124.14',
            'car_name' : 'Audi TTCup',
            'course_name' : 'Tokyo Central Outer' ,
            'num_cars' : 1,
            'spectator_mode' : False,
        })
        return super()._default_hparams().overwrite(default_dict)

    def _game_hp(self):
        game_hp = ParamDict({
            'builtin_controlled' : [],
            'do_init' : True,
            'reward_function' : reward_function,
            'done_function' : sampling_done_function,
            'standardize_observations' : False,
            'state_standard': True,
            'action_standard':False,
            'fast_mode':True,
            'min_frames_per_action': 6,
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
            min_frames_per_action = self._hp.min_frames_per_action, 
            feature_keys = RL_OBS_1, 
            # feature_keys = DEFAULT_FEATURE_KEYS,

            fast_mode = self._hp.fast_mode,
            builtin_controlled = self._hp.builtin_controlled, 
            spectator_mode = self._hp.spectator_mode,
            reward_function = self._hp.reward_function,
            done_function = self._hp.done_function,
            standardize_observations=self._hp.standardize_observations,
        )

        self.course_length = self._get_course_length()

    def reset(self, start_conditions=None):
        if not start_conditions:
            course_v = self.course_length * np.random.rand()
            start_conditions = start_condition_formulator(num_cars=1, course_v=[course_v], speed=[144])
        obs = self._env.reset(start_conditions=start_conditions)
        return self._wrap_observation(obs[0])

    def step(self, actions):
        actions = self.descaler_actions([actions])
        obs, rew, done, info = self._env.step(actions)
        return self._wrap_observation(obs[0]), rew[0], done[0], info

    def _wrap_observation(self, obs):
        converted_obs = raw_observation_to_true_observation(obs)
        if self.state_scaler:
            # print("scale obs")
            std_obs = self.state_scaler.transform([converted_obs])[0]
        else:
            std_obs = converted_obs
        return super()._wrap_observation(std_obs)

    def _get_course_length(self):
        course_length, course_code, course_name = self._env.get_course_meta()
        return course_length

    def render(self, mode='rgb_array'):
        return [[[0,0,0]]]

    def descaler_actions(self, actions):
        if self._hp.action_standard:
            return self.action_scaler.inverse_transform(actions)
        else:
            # print('not using standard actions')
            return actions

    def _load_standard_table(self):
        from spirl.utils.gts_utils import load_standard_table
        self.state_scaler, self.action_scaler = load_standard_table()

if __name__ == "__main__":
    from spirl.utils.general_utils import AttrDict
    conf = AttrDict({'do_init' : True})
    # conf = AttrDict({'do_init' : False})
    env  = GTSEnv_Base(conf)
    obs = env.reset()
    obs, rew, done, info = env.step([0, -1])
    print('obs shape', obs.shape)
    print('rew shape', rew)
    print('done shape', done)


    # python spirl/rl/envs/gts.py