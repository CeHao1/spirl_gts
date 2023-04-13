
from spirl.rl.components.sampler_batched import AgentDetached_HierarchicalSamplerBatched

from spirl.utils.general_utils import listdict2dictlist, batch_listdict2dictlist, AttrDict, ParamDict, obj2np
import torch.multiprocessing as mp
# import multiprocessing as mp

class SamplerWrapped:
    def __init__(self, config, env, agent, logger, max_episode_len):
        self._hp = self._default_hparams().overwrite(config)
        
        # if only set logger for the master sampler
        self._sub_samplers = []
        for i in range(len(env)):
            logger_i = logger if i==0 else None
            self._sub_samplers.append(self._hp.sub_sampler(config, env[i], agent, logger_i, max_episode_len))
        self.num_envs = len(self._sub_samplers)
        
        mp.set_start_method('spawn')
    
    def _default_hparams(self):
        return ParamDict({
            'sub_sampler': AgentDetached_HierarchicalSamplerBatched,
        })

    def init(self, is_train):
        for rank in range(self.num_envs):
            self._sub_samplers[rank].init(is_train)


    def sample_batch(self, batch_size, is_train=True, global_step=None, store_ll=True):
        # modify batch_size
        import math
        batch_size_every = math.ceil(batch_size / self.num_envs)
        with mp.Pool(processes=self.num_envs) as pool:
            
            results = [pool.apply_async(self._sub_samplers[i].sample_batch, (batch_size_every, is_train, global_step)) for i in range(self.num_envs)]
            results = [p.get() for p in results]
            
        return self._process_sample_batch_return(results)

    def sample_episode(self, is_train, render=False, deterministic_action=False, return_list=False):
        # multi processing
        with mp.Pool(processes=self.num_envs) as pool:
            results = [pool.apply_async(self._sub_samplers[i].sample_episode, (is_train, render, deterministic_action, return_list)) for i in range(self.num_envs)]
            results = [p.get() for p in results]

        return self._process_sample_episode_return(results)
        

    def _process_sample_batch_return(self, results):
        experience_batch_list = []
        env_steps_sum = 0
        for result in results:
            experience_batch, env_steps = result
            experience_batch_list.append(experience_batch)
            env_steps_sum += env_steps

        return batch_listdict2dictlist(experience_batch_list), env_steps_sum

    def _process_sample_episode_return(self, results):
        return batch_listdict2dictlist(results)

class HierarchicalSamplerWrapped(SamplerWrapped):
    pass