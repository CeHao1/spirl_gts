from gym_gts import GTSApi
import gym

from spirl.utils.general_utils import ParamDict

from spirl.utils.gts_utils import DEFAULT_FEATURE_KEYS
 

class BaseSampler:
    def __init__(self, config):
        self._hp = self._default_hparams().overwrite(config)
        self._make_env()

    def _default_hparams(self):
        default_dict = ParamDict({
            'ip_address':               '',
            'builtin_controlled':       [],
            'min_frames_per_action':    1,
            'spectator_mode':           True,
            'store_states':             False,
            'standardize_observations': False,
        })
        return default_dict

    def _make_env(self):
        self.env = gym.make('gts-v0', 
        
            ip = self._hp.ip_address, 
            feature_keys = DEFAULT_FEATURE_KEYS,
            min_frames_per_action = self._hp.min_frames_per_action, 
            # reward_function = self._hp.reward_function,
            # done_function = self._hp.done_function,
            builtin_controlled = self._hp.builtin_controlled,
            store_states = self._hp.store_states,
            standardize_observations=self._hp.standardize_observations,
            )

    def sample(self, start_conditions):
        self.env.reset_spectator(start_conditions = start_conditions)

        for idx in range(round(1e15)):

            # observe
            gts_state_library = self.env.observe_states()
            

            # calculate time, lap_count, course_v
            now_frame_count = gts_state_library[0]['frame_count']

            # check 