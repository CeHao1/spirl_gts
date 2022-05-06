from gym_gts import GTSApi
import gym

import numpy as np

from spirl.rl.components.environment import GymEnv
from spirl.rl.envs.gts import GTSEnv_Base
from spirl.utils.general_utils import ParamDict, AttrDict

from spirl.utils.gts_utils import make_env, initialize_gts
from spirl.utils.gts_utils import RL_OBS_1, CAR_CODE, COURSE_CODE, TIRE_TYPE, BOP
from spirl.utils.gts_utils import raw_observation_to_true_observation

from spirl.utils.gts_utils import reward_function, sampling_done_function

class GTSEnv_Multi(GTSEnv_Base):

# we can run 20 cars at the same time, but we need to separate the trajectory, 
# just like sampler

    def _game_hp(self):
        game_hp = ParamDict({
            'builtin_controlled' : [],
            'do_init' : False,
            'reward_function' : reward_function,
            'done_function' : sampling_done_function
        })
        return game_hp

    def _default_hparams(self):
        default_dict = ParamDict({
            'ip_address' : '192.168.124.14',
            'car_name' : 'Audi TTCup',
            'course_name' : 'Tokyo Central Outer' ,
            'num_cars' : 20,
            'spectator_mode' : False,
        })
        return super()._default_hparams().overwrite(default_dict)

    def reset(self, start_conditions=None):
        obs = self._env.reset(start_conditions=start_conditions)
        return self._wrap_observation(obs)

    def step(self, actions):
        obs, rew, done, info = self._env.step(actions)
        return self._wrap_observation(obs), rew, done, info

    def _wrap_observation(self, obs):
        converted_obs = [raw_observation_to_true_observation(obs_single) for obs_single in obs]
        return GymEnv._wrap_observation(self, converted_obs) 

    def render(self, mode='rgb_array'):
        return [[0,0,0] for _ in range(self._hp.num_cars)]


if __name__ == "__main__":
    from spirl.utils.general_utils import AttrDict
    conf = AttrDict({'do_init' : True})
    # conf = AttrDict({'do_init' : False})
    env  = GTSEnv_Multi(conf)
    obs = env.reset()
    obs, rew, done, info = env.step([0, -1])
    print('obs shape', obs.shape)
    print('rew shape', rew)
    print('done shape', done)


    # python spirl/rl/envs/gts_multi.py

    

        

    