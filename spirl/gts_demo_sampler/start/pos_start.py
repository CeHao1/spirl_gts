from spirl.utils.general_utils import ParamDict

from spirl.gts_demo_sampler.start.base_start import BaseStart

class PosStart(BaseStart):
    def __init__(self, config):
        self._hp = self._default_hparams().overwrite(config)

    def _default_hparams(self):
        default_dict = ParamDict({
            'num_cars' :        1,
            'pos' :             [[0,0,0]],
            'rot' :             [[0,0,0]],
            'speed_kmph':       [144],
        })
        return super()._default_hparams().overwrite(default_dict)

    def _generate_condition(self, idx):
        condition = {
            'id': idx,
            'pos': self._hp.pos[idx],
            'rot': self._hp.rot[idx],
            'speed_kmph': self._hp.speed_kmph[idx],
        }
        return condition