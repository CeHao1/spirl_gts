from gym_gts import GTSApi
import gym
import numpy as np

from spirl.utils.general_utils import ParamDict

from spirl.utils.gts_utils import DEFAULT_FEATURE_KEYS
 

class BaseSample:
    def __init__(self, config):
        self._hp = self._default_hparams().overwrite(config)
        # self._make_env()

    def _default_hparams(self):
        default_dict = ParamDict({
            'ip_address':               '',
            'builtin_controlled':       [],
            'min_frames_per_action':    1,
            'spectator_mode':           True,
            'store_states':             False,
            'standardize_observations': True,
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
            spectator_mode = self._hp.spectator_mode,
            )

    def sample_raw_data(self, start_conditions, done_function):
        self._make_env()
        self.env.reset_spectator(start_conditions = start_conditions)

        raw_state_list = []
        expected_frame_count = -1

        for idx in range(round(1e15)):

            # observe
            gts_state_list = self.env.observe_states()
            

            # calculate time, lap_count, course_v
            now_frame_count = gts_state_list[0]['frame_count']

            # check next time step
            if ( now_frame_count >= expected_frame_count):
                raw_state_list.append(gts_state_list)
                expected_frame_count = now_frame_count + self._hp.min_frames_per_action

            # check done
            done_feature = self._get_done_feature(gts_state_list)
            if done_function(done_feature):
                # self.env.close()
                break


        # now the data is like data = [t0, t1, ...], t0=[car0, car1, ...], car0 = dict('state0, state1, ...')
        # we need to make data = [c0, c1, ,,,], c0=[t0, t1, ...], to = dict(state0, state1, ...)
        # this is a transpose process
        raw_state_list = np.array(raw_state_list).transpose()

        return raw_state_list

    def _get_done_feature(self, state_list):
        'time, lap_count, course_v'
        time = [state['frame_count']/60 for state in state_list]
        lap_count = [state['lap_count'] for state in state_list]
        course_v = [state['course_v'] for state in state_list]

        return {'time':time, 'lap_count':lap_count, 'course_v':course_v}