
from spirl.rl.components.sampler_batched import AgentDetached_HierarchicalSamplerBatched

from spirl.utils.general_utils import listdict2dictlist, batch_listdict2dictlist, listdict_mean
from spirl.utils.general_utils import AttrDict, ParamDict, obj2np
import torch.multiprocessing as mp
# import multiprocessing as mp

class SamplerWrapped:
    def __init__(self, config, env, agent, logger, max_episode_len):
        self._hp = self._default_hparams().overwrite(config)
        self._logger = logger
        self._sub_samplers = [self._hp.sub_sampler(config, env_i, agent, logger, max_episode_len) for env_i in env]
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
        
        experience_batch, env_step, episode_info = self._process_sample_batch_return(results)
        
        if global_step is not None:
            self._logger.log_scalar_dict(episode_info, prefix='train' if is_train else 'val', 
                                         step=global_step)
        
        return experience_batch, env_step

    def sample_episode(self, is_train, render=False, deterministic_action=False):
        # multi processing
        with mp.Pool(processes=self.num_envs) as pool:
            results = [pool.apply_async(self._sub_samplers[i].sample_episode, (is_train, render, deterministic_action)) for i in range(self.num_envs)]
            results = [p.get() for p in results]

        return self._process_sample_episode_return(results)

    def _process_sample_batch_return(self, results):
        experience_batch_list = []
        episode_info_list = []
        env_steps_sum = 0
        for result in results:
            experience_batch, env_steps, episode_info = result
            experience_batch_list.append(experience_batch)
            episode_info_list.append(episode_info)
            env_steps_sum += env_steps

        episode_info = listdict_mean(episode_info_list)
        return batch_listdict2dictlist(experience_batch_list), env_steps_sum, episode_info

    def _process_sample_episode_return(self, results):
        return batch_listdict2dictlist(results)

class HierarchicalSamplerWrapped(SamplerWrapped):
    def _process_sample_batch_return(self, results):
        hl_experience_batch_list = []
        ll_experience_batch_list = []
        episode_info_list = []
        env_steps_sum = 0
        for result in results:
            experience_batch, env_steps, episode_info = result
            hl_experience_batch_list.append(experience_batch.hl_batch)
            ll_experience_batch_list.append(experience_batch.ll_batch)
            episode_info_list.append(episode_info)
            env_steps_sum += env_steps
        
        episode_info = listdict_mean(episode_info_list)
        return AttrDict(hl_batch = batch_listdict2dictlist(hl_experience_batch_list),
                        ll_batch = batch_listdict2dictlist(ll_experience_batch_list)), \
                env_steps_sum, episode_info