
from spirl.utils.general_utils import ParamDict

from spirl.utils.gts_utils import CAR_CODE, COURSE_CODE, TIRE_TYPE, BOP
from spirl.utils.gts_utils import make_env, initialize_gts

import copy

class BaseInit:
    def __init__(self, config):
        self._hp = self._default_hparams().overwrite(config)
        self._generate_bop()

    def _default_hparams(self):
        default_dict = ParamDict({
            'ip_address'    : None,
            'num_cars'      : 1,
            'car_name'      : 'Audi TTCup',
            'course_name'   : 'Tokyo Central Outer',
            'tire_type'     : 'RH',
            'bop'           : None,                 # this could be a list of percentage [[0.9, 1.1], [xx, xx]] power, weights
        })
        return default_dict

    def _generate_bop(self):
        if self._hp.bop is None:
            self.bops = [BOP[self._hp.car_name]] * self._hp.num_cars
        else:
            assert len(self._hp.bop) == self._hp.num_cars
            bop_template = BOP[self._hp.car_name]
            self.bops = []

            for idx in range(self._hp.num_cars):
                bop = copy.deepcopy(bop_template)
                bop['power'] = round(bop['power'] * self._hp.bop[idx][0])
                bop['weight'] = round(bop['weight'] * self._hp.bop[idx][1])
                self.bops.append(bop)

    def init_gts(self):
        # from gym_gts import GTSApi
        # with GTSApi(ip= self._hp.ip_address) as gts_api:
        #     gts_api.set_race(
        #         num_cars = self._hp.num_cars,
        #         car_codes = CAR_CODE[self._hp.car_name],
        #         course_code = COURSE_CODE[self._hp.course_name],
        #         front_tires = self._hp.tire_type,
        #         rear_tires = self._hp.tire_type,
        #         bops = self.bops
        #     )

        initialize_gts(
            ip =self._hp.ip_address, 
            num_cars = self._hp.num_cars, 
            car_codes = CAR_CODE[self._hp.car_name], 
            course_code = COURSE_CODE[self._hp.course_name], 
            tire_type = self._hp.tire_type, 
            bops = self.bops
        )



if __name__ == '__main__':
    pass