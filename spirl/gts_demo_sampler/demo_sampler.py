from gym_gts import GTSApi

from utils.gts_utils import BoP_formulator, initialize_gts\
from gts_demo_sampler.param import get_args

from enum import Enum

'''
In this function, we set the initial condition and save raw dict data.
We can sample 20 cars or 1 car by initialization. specified by the id.
'''

# we defines some demonstration_sample_mode
'''
1. Whole lap with limit time, time-trial
2. Chosen course_v, time-trial
3. With a leader car, versus
'''
class demo_sampler_mode(Enum):
    TIME_TRIAL_WHOLE_LAP = 0
    TIME_TRIAL_CORNER_2 = 1
    VERSUS_CORNER_2 = 2


# initialization method
'''
1. initialize by course_v in [min, max]
2. initialize by pos / rot 
'''
 

class DemoSampler:
    def __init__(self, args):
        self._hp = args

        self._hp.exp_path = make_path(self.conf.exp_dir, args.path, args.prefix, args.new_dir)


    def get_start_conditions(self):
        pass

    def sample_framework(self, start_conditions):
        pass

    def reset_by_start_conditions(self):
        pass

    def save_raw_data_dict(self, raw_data_dict_list):
        pass

    def convert_raw_data_to_rollout(self):
        pass



if __name__ == "__main__":
    DemoSampler(args=get_args())