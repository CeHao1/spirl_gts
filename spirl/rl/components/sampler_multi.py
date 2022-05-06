import numpy as np
import contextlib

from spirl.rl.components.sampler import Sampler
from spirl.utils.general_utils import listdict2dictlist, AttrDict, ParamDict, obj2np

class SamplerMulti(Sampler):

    # config must have a argument: number_of_agents

    # def sample_action(self, obs):
    #     actions = [self._agent.act(obs_single) for obs_single in obs]
    #     return actions

    def sample_batch(self, batch_size, is_train=True, global_step=None):
        na = self._hp.number_of_agents

        experience_batch = [[] for _ in range(na)]
        step = 0
        with self._env.val_mode() if not is_train else contextlib.suppress():
            with self._agent.val_mode() if not is_train else contextlib.suppress():
                with self._agent.rollout_mode():
                    while step < batch_size:
                        agent_output = [self.sample_action(self._obs[agent_index]) for agent_index in range(na)]
                        if agent_output[0].action is None:
                            self._episode_reset(global_step)
                            continue

                        actions = [agent_output[agent_index].action for agent_index in range(na)]
                        obs, reward, done, info = self._env.step(actions)

                        for agent_index in range(na): 
                            experience_batch[agent_index].append(AttrDict(
                                observation=self._obs[agent_index],
                                reward=reward[agent_index],
                                done=done[agent_index],
                                action=agent_output[agent_index].action,
                                observation_next=obs[agent_index],
                            ))

                            self._episode_reward += reward[agent_index]

                        self._obs = obs
                        step += na
                        self._episode_step += na

                        # reset if episode ends
                        if np.any(done) or self._episode_step >= self._max_episode_len:
                            if not np.all(done):    # force done to be True for timeout
                                for agent_index in range(na): 
                                    experience_batch[agent_index][-1].done = True
                            self._episode_reset(global_step)

        experience_batch_final = []
        for agent_index in range(na): 
            experience_batch_final += experience_batch[agent_index]
        return listdict2dictlist(experience_batch_final), step

    def sample_episode(self, is_train, render=False):
        na = self._hp.number_of_agents

        self.init(is_train)
        episode = [[] for _ in range(na)]
        done = [False] * na

        with self._env.val_mode() if not is_train else contextlib.suppress():
            with self._agent.val_mode() if not is_train else contextlib.suppress():
                with self._agent.rollout_mode():
                    while not np.any(done) and self._episode_step < self._max_episode_len:
                        agent_output = [self.sample_action(self._obs[agent_index]) for agent_index in range(na)]

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

        episode_final = []
        for agent_index in range(na): 
            episode_final += episode[agent_index]
        return listdict2dictlist(episode_final)

