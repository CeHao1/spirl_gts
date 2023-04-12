
from spirl.utils.general_utils import ParamDict

class EnvList:
    def __init__(self, config):
        self._hp = self._default_hparams().overwrite(config)
        self._envs = [self._make_env(env_config) for env_config in self._hp.sub_env_configs]

    def _default_hparams(self):
        return ParamDict({
            'sub_env_configs': [],
        })

    def __iter__(self):
        return iter(self._envs)