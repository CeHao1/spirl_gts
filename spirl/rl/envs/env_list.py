
from spirl.utils.general_utils import ParamDict
from spirl.rl.components.environment import BaseEnvironment

class EnvList(BaseEnvironment):
    def __init__(self, config):
        self._hp = self._default_hparams().overwrite(config)
        self._envs = [self._hp.env_class(env_config) for env_config in self._hp.sub_env_configs]

    def _default_hparams(self):
        return ParamDict({
            'env_class': None,
            'sub_env_configs': [],
        })

    def __iter__(self):
        return iter(self._envs)
    
    def __getitem__(self, idx):
        return self._envs[idx]
    
    def __len__(self):
        return len(self._envs)