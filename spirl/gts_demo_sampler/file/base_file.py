
from spirl.utils.general_utils import ParamDict

'''
pickle, 
'''

class BaseFile:
    def __init__(self, config):
        self._hp = self._default_hparams().overwrite(config)

    def _default_hparams(self):
        default_dict = ParamDict({
            'raw_data_dir'      : '',
            'rollout_dr'        : '',
            'feature_keys'      : None,
            'action_keys'       : None,
        })
        return default_dict 

    def save_raw_data(self, raw_data_list):
        pass

        # name of file

        # path of file

        # save 


    def convert_to_rollout(self):
        pass

        # get all file names in the dir

        # convert to obs, next_obs

        # convert to action,

        # generate dummpy image,

        # save as rollout