
from spirl.rl.components.sampler_batched import AgentDetached_HierarchicalSamplerBached

from spirl.utils.general_utils import listdict2dictlist, AttrDict, ParamDict, obj2np
# import torch.multiprocessing as mp
import multiprocessing as mp

class SamplerWrapped:
    def __init__(self, config, env, agent, logger, max_episode_len):
        self._hp = self._default_hparams().overwrite(config)
        self._sub_samplers = [self._hp.sub_sampler(config, env_i, agent, logger, max_episode_len) for env_i in env]
        pass
    
    def _default_hparams(self):
        return ParamDict({
            'sub_sampler': AgentDetached_HierarchicalSamplerBached,
        })

    def init(self, is_train):
        for rank in range(len(self._sub_samplers)):
            self._sub_samplers[rank].init(is_train)

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

        return self._process_sample_batch_return(results)

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

        return self._process_sample_episode_return(results)

    def _process_sample_batch_return(self, results):
        experience_batch_list = []
        env_steps_sum = 0
        for result in results:
            experience_batch, env_steps = result
            experience_batch_list.append(experience_batch)
            env_steps_sum += env_steps

        return listdict2dictlist(experience_batch_list), env_steps_sum

    def _process_sample_episode_return(self, results):
        return listdict2dictlist(results)

class HierarchicalSamplerWrapped(SamplerWrapped):
    pass