
from spirl.rl.components.sampler_batched import AgentDetached_HierarchicalSamplerBached

from spirl.utils.general_utils import listdict2dictlist, AttrDict, ParamDict, obj2np
import torch.multiprocessing as mp

class HierarhicalSamplerWrapped:
    def __init__(self, config, env, agent, logger, max_episode_len):
        self._hp = self._default_hparams().overwrite(config)
        self._sub_samplers = [(self, config, env_i, agent, logger, max_episode_len) for env_i in env]
    
    def _default_hparams(self):
        return ParamDict({
            'sub_sampler': AgentDetached_HierarchicalSamplerBached,
        })

    def sample_batch(self, batch_size, is_train=True, global_step=None, store_ll=True):
        # multi processing
        Q = mp.Queue()

        processes = []
        for rank in range(len(self._sub_samplers)):
            p = mp.Process(target=self._sub_samplers[rank].sample_batch, 
                          args=(batch_size, is_train, global_step, store_ll, Q))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
        # collect results
        results = []
        while not Q.empty():
            results.append(Q.get())

        return listdict2dictlist(results)

    def sample_episode(self, is_train, render=False, deterministic_action=False, return_list=False):
        # multi processing
        Q = mp.Queue()

        processes = []
        for rank in range(len(self._sub_samplers)):
            p = mp.Process(target=self._sub_samplers[rank].sample_episode, 
                          args=(is_train, render, deterministic_action, return_list, Q))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

        # collect results
        results = []
        while not Q.empty():
            results.append(Q.get())

        return listdict2dictlist(results)


