
from spirl.utils.general_utils import ParamDict, AttrDict
from spirl.utils.gts_utils import chosen_feature_keys, action_keys
from spirl.gts_demo_sampler.file.file_operation import *

import pickle
import os
from tqdm import tqdm

class BaseFile:
    def __init__(self, config):
        self._hp = self._default_hparams().overwrite(config)

    def _default_hparams(self):
        default_dict = ParamDict({
            'raw_data_dir'      : './sample/raw_data',
            'rollout_dir'       : './sample/rollout',
            'feature_keys'      : chosen_feature_keys,
            'action_keys'       : action_keys,
            'save_car_id'       : [0]  # which car to save
        })
        return default_dict 

    def save_raw_data(self, raw_data_list, file_name):
        make_dir(self._hp.raw_data_dir)

        # name of file
        for idx in range(len(raw_data_list)):
            if idx not in self._hp.save_car_id: # skip
                continue
            file_dir = os.join(self._hp.raw_data_dir, file_name + '_car' + str(idx))
            save_file(file_dir, raw_data_list[idx])


    def convert_to_rollout(self):
        file_names = os.listdir(self._hp.raw_data_dir)
        for idx in tqdm(range(len(file_names))):
            file_dir = os.join(self._hp.raw_data_dir, file_names[idx])
            state_one_car = load_file(file_dir)
            ep = self.formulate_episode(state_one_car)
            save_rollout(str(idx), ep, self._hp.rollout_dir)
        
    def formulate_episode(self, states_one_car):
        observation = []
        for key in self._hp.feature_keys:
            # if key in ['t', 's']:
            observation.append(states_one_car[key])

        action = []
        steer2range = np.pi / 6
        action.append(np.array(states_one_car['steering']) / steer2range)
        action.append(np.array(states_one_car['throttle']) - np.array(states_one_car['brake']))

        observation = np.array(observation).T
        action = np.array(action).T

        image = np.zeros((len(states_one_car['frame_count']), 1, 1, 3))
        done = [False for _ in image]
        done[-1] = True
        done = np.array(done)
        episode = AttrDict( observation=observation, action=action , image = image, done=done)

        # print(observation.shape, action.shape, image.shape)
        return episode