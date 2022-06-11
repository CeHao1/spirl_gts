import numpy as np
import contextlib

from spirl.rl.components.sampler import Sampler, HierarchicalSampler
from spirl.modules.variational_inference import MultivariateGaussian
from spirl.utils.general_utils import listdict2dictlist, AttrDict, ParamDict, obj2np
from copy import deepcopy
from time import time

class SamplerMulti(Sampler):

    def sample_batch(self, batch_size, is_train=True, global_step=None):
        na = self._hp.number_of_agents
        experience_batch = [[] for _ in range(na)]
        step = 0
        with self._env.val_mode() if not is_train else contextlib.suppress():
            with self._agent.val_mode() if not is_train else contextlib.suppress():
                with self._agent.rollout_mode():
                    while step < batch_size:
                        # print('\n===========================')
                        t0 = time()
                        
                        agent_output = [self.sample_action(self._obs[agent_index]) for agent_index in range(na)]

                        if agent_output[0].action is None:
                            self._episode_reset(global_step)
                            continue

                        actions = [agent_output[agent_index].action for agent_index in range(na)]
                        # actions = [[0.0, 0.0] for agent_index in range(na)]
                        
                        # t1 = time()
                        # print('time to generate action', (t1-t0)*1000)


                        obs, reward, done, info = self._env.step(actions)
                        # t2 = time()
                        # print('time for step', (t2-t1)*1000)

                        for agent_index in range(na): 
                            experience_batch[agent_index].append(AttrDict(
                                observation=self._obs[agent_index],
                                reward=reward[agent_index],
                                done=done[agent_index],
                                action=agent_output[agent_index].action,
                                observation_next=obs[agent_index],
                            ))

                            self._episode_reward += reward[agent_index] / na # average reward of one car

                        self._obs = obs
                        step += na
                        self._episode_step += na

                        # reset if episode ends
                        if np.any(done) : # must sample the complete trajectory
                        # or self._episode_step >= self._max_episode_len:
                            if not np.all(done):    # force done to be True for timeout
                                for agent_index in range(na): 
                                    experience_batch[agent_index][-1].done = True
                            self._episode_reset(global_step)

                        # t3 = time()
                        # print('time for rest', (t3-t2)*1000)

        experience_batch_final = []
        for agent_index in range(na): 
            experience_batch_final += experience_batch[agent_index]
        return listdict2dictlist(experience_batch_final), step

    def sample_episode(self, is_train, render=False, deterministic_action=False, return_list=False):
        na = self._hp.number_of_agents

        self.init(is_train)
        episode = [[] for _ in range(na)]
        done = [False] * na

        with self._env.val_mode() if not is_train else contextlib.suppress():
            with self._agent.val_mode() if not is_train else contextlib.suppress():
                with self._agent.rollout_mode():
                    while not np.all(done) :
                    # and self._episode_step < self._max_episode_len:
                        agent_output = [self.sample_action(self._obs[agent_index]) for agent_index in range(na)]
                        agent_output = self._postprocess_agent_output(agent_output, deterministic_action=deterministic_action)
                        if render:
                            render_obs = self._env.render()

                        actions = [agent_output[agent_index].action for agent_index in range(na)]
                        obs, reward, done, info = self._env.step(actions)

                        for agent_index in range(na): 
                            episode[agent_index].append(AttrDict(
                                observation=self._obs[agent_index],
                                reward=reward[agent_index],
                                done=done[agent_index],
                                action=agent_output[agent_index].action,
                                observation_next=obs[agent_index],
                            ))
                            if render:
                                episode[agent_index][-1].update(AttrDict(image=render_obs[agent_index]))
                            
                        self._obs = obs
                        self._episode_step += na

        for agent_index in range(na): 
            episode[agent_index][-1].done = True

        if return_list:
            return [listdict2dictlist(one_epsi) for one_epsi in episode]

        episode_final = []
        for agent_index in range(na): 
            episode_final += episode[agent_index]
        return listdict2dictlist(episode_final)


    def _postprocess_agent_output(self, agent_output, deterministic_action=False):
        if deterministic_action:
            if isinstance(agent_output[0].dist, MultivariateGaussian):
                    for idx in range(len(agent_output)):
                        agent_output[idx].ori_action = agent_output[idx].action
                        agent_output[idx].action = agent_output[idx].dist.mean[0]
                        # clip to [-1, 1]
                        agent_output[idx].action = np.clip(agent_output[idx].action, -1.0, 1.0)
        return agent_output


class HierarchicalSamplerMulti(SamplerMulti, HierarchicalSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_hl_obs = [None for _ in range(self._hp.number_of_agents)]
        self.last_hl_action = [None for _ in range(self._hp.number_of_agents)]  # stores observation when last hl action was taken
        self.reward_since_last_hl = [0] * self._hp.number_of_agents  # accumulates the reward since the last HL step for HL transition
        self._obs = [None for _ in range(self._hp.number_of_agents)]
        # self._episode_reward = [0] * self._hp.number_of_agents


    def sample_batch(self, batch_size, is_train=True, global_step=None, store_ll=True):
        """Samples the required number of high-level transitions. Number of LL transitions can be higher."""
        na = self._hp.number_of_agents

        hl_experience_batch = [[] for _ in range(na)]
        ll_experience_batch = [[] for _ in range(na)]

  
        env_steps, hl_step = 0, 0
        with self._env.val_mode() if not is_train else contextlib.suppress():
            with self._agent.val_mode() if not is_train else contextlib.suppress():
                with self._agent.rollout_mode():
                    # copy 20 agents for multi-sampling, only temprary variables
                    multi_agent = [deepcopy(self._agent) for _ in range(na)]

                    while hl_step < batch_size or len(ll_experience_batch[0]) <= 1:
                        # print('\n==============================================')
                        
                        # initially reset for agents
                        if self.initial_reset_counts < na:
                            multi_agent[self.initial_reset_counts].reset()
                            
                            # print('reset agent', self.initial_reset_counts)
                        

                        # perform one rollout step
                        # t0 = time()
                        agent_output = [multi_agent[agent_index].act(self._obs[agent_index]) for agent_index in range(na)]
                        # t1 = time()
                        # print('** get action time is ', (t1 - t0)*1000)

                        
                        actions = [agent_output[agent_index].action for agent_index in range(na)]

                        obs, reward, done, info = self._env.step(actions)
                        # obs = self._postprocess_obs(obs)

                        for agent_index in range(na): 

                            # store low-level observations, usually not useful
                            if store_ll:
                                if ll_experience_batch[agent_index]:
                                    ll_experience_batch[agent_index][-1].observation_next = \
                                        self._agent.make_ll_obs(ll_experience_batch[agent_index][-1].observation_next, agent_output[agent_index].hl_action)
                                
                                # store current step in ll_experience_batch
                                ll_experience_batch[agent_index].append(AttrDict(
                                    observation=self._agent.make_ll_obs(self._obs[agent_index], agent_output[agent_index].hl_action),
                                    reward=reward[agent_index],
                                    done=done[agent_index],
                                    action=agent_output[agent_index].action,
                                    observation_next=obs[agent_index],       # this will get updated in the next step
                                ))


                            # store high-level actions
                            if agent_output[agent_index].is_hl_step or np.any(done) or (self._episode_step >= self._max_episode_len-1):
                                if self.last_hl_obs[agent_index] is not None and self.last_hl_action[agent_index] is not None:
                                    # do not store until several steps, the initial steps are very unstable
                                    if (self.initial_reset_counts < na * 3):
                                        break
                                    
                                    hl_experience_batch[agent_index].append(AttrDict(
                                        observation=self.last_hl_obs[agent_index],
                                        reward=self.reward_since_last_hl[agent_index],
                                        done=done[agent_index],
                                        action=self.last_hl_action[agent_index],
                                        observation_next=obs[agent_index],
                                    ))
                                    hl_step += 1

                                    # print('==== high-level step for agent', agent_index)

                                if np.any(done):
                                    hl_experience_batch[agent_index][-1].reward += reward[agent_index]  # add terminal reward

                                self.last_hl_obs[agent_index] = self._obs[agent_index] if self._episode_step == 0 else obs[agent_index]
                                self.last_hl_action[agent_index] = agent_output[agent_index].hl_action
                                self.reward_since_last_hl[agent_index] = 0

                            
                            # self._obs[agent_index] = obs[agent_index]
                            env_steps += 1
                            self._episode_step += 1

                            # _episode_reward must a number, average of each car
                            self._episode_reward += reward[agent_index] / na
                            self.reward_since_last_hl[agent_index] += reward[agent_index]

                        # update stored observation, what ever
                        self._obs = obs
                        self.initial_reset_counts += 1

                        # reset if episode ends
                        if np.any(done) or self._episode_step >= self._max_episode_len:
                            if not np.all(done):    # force done to be True for timeout
                                for agent_index in range(na): 
                                    ll_experience_batch[agent_index][-1].done = True
                                    if hl_experience_batch[agent_index]:   # can potentially be empty 
                                        hl_experience_batch[agent_index][-1].done = True
                            self._episode_reset(global_step)

                        # for agent numbers

                    # end while

        final_hl_experience_batch = []
        final_ll_experience_batch = []

        for agent_index in range(na):
            final_hl_experience_batch += hl_experience_batch[agent_index]
            final_ll_experience_batch += ll_experience_batch[agent_index][:-1]

        return AttrDict(
            hl_batch=listdict2dictlist(final_hl_experience_batch),
            ll_batch=listdict2dictlist(final_ll_experience_batch),   # last element does not have updated obs_next!
        ), env_steps


    def _episode_reset(self, global_step=None):
        Sampler._episode_reset(self, global_step)
        self.last_hl_obs = [None for _ in range(self._hp.number_of_agents)]
        self.last_hl_action = [None for _ in range(self._hp.number_of_agents)]  # stores observation when last hl action was taken
        self.reward_since_last_hl = [0] * self._hp.number_of_agents  # accumulates the reward since the last HL step for HL transition
        # self._episode_reward = [0] * self._hp.number_of_agents

        self.initial_reset_counts = 0 
        # because sampling high-level action takes too long time, if we sample for 20 cars, 
        # the time is too long and will cause delay. So, we will initially separate them.