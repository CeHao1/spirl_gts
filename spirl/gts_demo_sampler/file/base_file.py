
from spirl.utils.general_utils import ParamDict
from spirl.utils.gts_utils import chosen_feature_keys, action_keys

import pickle
import os

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
        print('got it!')
        pass

        # name of file

        # path of file

        # save 


    def convert_to_rollout(self):
        print('nothing!')
        pass

        # get all file names in the dir

        # convert to obs, next_obs

        # convert to action,

        # generate dummpy image,

        # save as rollout