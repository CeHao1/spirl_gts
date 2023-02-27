
from spirl.utils.general_utils import ParamDict

'''
In this function, we can input some conditions, then generate the corresponding state_condition dict
'''

'''
{"launch": conditions}
conditions = [c0, c1, ...]
c1 = {'id': 1, } 
course_v: 2000, 
speed_kmph: 144,

'''


class BaseStart:
    def __init__(self, config):
        self._hp = self._default_hparams().overwrite(config)
        self.generate_start_conditions()

    def _default_hparams(self):
        default_dict = ParamDict({
            'num_cars' :        1,
            'course_v' :        [0],
            'speed_kmph':       [144],
        })
        return default_dict 

    def generate_start_conditions(self):
        conditions = []
        for car_index in range(self._hp.num_cars):
            condition = self._generate_condition(car_index)
            conditions.append(condition)
        self._start_conditions = {"launch": conditions}

    def _generate_condition(self, idx):
        condition = {
            'id': idx,
            'course_v': self._hp.course_v[idx],
            'speed_kmph': self._hp.speed_kmph[idx],
        }
        return condition

    @property
    def start_conditions(self):
        self.generate_start_conditions()
        return self._start_conditions

def convert_pos_rot(course_v, epsi, ey):
    pass

if __name__ == "__main__":
    config = ParamDict()
    start = BaseStart(config)