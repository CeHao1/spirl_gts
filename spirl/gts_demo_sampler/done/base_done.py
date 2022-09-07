
from spirl.utils.general_utils import ParamDict

import numpy as np


class BaseDone:
    def __init__(self, config):
        self._hp = self._default_hparams().overwrite(config)
        self._done_function = self._generate_combined_done_function()

    def _default_hparams(self):
        # e.g. max_time = [100, 100] (seconds)
        default_dict = ParamDict({
            'max_time' :        None,
            'max_lap_count' :   None,
            'max_course_v':     None,
            'verbose':          False,
        })
        return default_dict 

    def _generate_combined_done_function(self):
        done_functions = self._generate_separate_done_functions()
        done_function = self._combined_done_functions(done_functions)
        return done_function
        
    def _generate_separate_done_functions(self):
        always_true_fun = lambda x : True
        max_time_fun = always_true_fun
        max_lap_count_fun = always_true_fun
        max_course_v_fun = always_true_fun

        if self._hp.max_time is not None:
            max_time_fun = max_time_generator(self._hp.max_time)
        if self._hp.max_lap_count is not None:
            max_lap_count_fun = max_lap_count_generator(self._hp.max_lap_count)
        if self._hp.max_course_v is not None:
            max_course_v_fun = max_course_v_generator(self._hp.max_course_v)

        return {'time':max_time_fun , 'lap_count':max_lap_count_fun, 'course_v':max_course_v_fun}

    def _combined_done_functions(self, done_function_list):
        def done_function(input_states):
            done = False
            for state_name in done_function_list:
                done = done or done_function_list[state_name](input_states[state_name])
            return done
        return done_function

    @property
    def done_function(self):
        return self._done_function


# ================== done function generator ====================
def max_time_generator(max_time):
    def max_time_fun(time : np.ndarray):
        return np.any(np.array(time) > np.array(max_time))
    return max_time_fun

def max_lap_count_generator(max_lap_count):
    def max_lap_count_fun(lap_count):
        return np.any(np.array(lap_count) > np.array(max_lap_count))
    return max_lap_count_fun

def max_course_v_generator(max_course_v):
    def max_course_v_fun(course_v):
        return np.any(np.array(course_v) > np.array(max_course_v))
    return max_course_v_fun



if __name__ == "__main__":
    from spirl.utils.general_utils import AttrDict
    config = AttrDict(
    max_course_v = 2000,
    max_time = 100,
    max_lap_count = 2,
)