import numpy as np
import contextlib

from spirl.rl.components.sampler import Sampler
from spirl.utils.general_utils import listdict2dictlist, AttrDict, ParamDict, obj2np

class SamplerMulti(Sampler):

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


class HierarchicalSamplerMulti(SamplerMulti):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_hl_obs, self.last_hl_action = None, None  # stores observation when last hl action was taken
        self.reward_since_last_hl = 0  # accumulates the reward since the last HL step for HL transition

    def sample_batch(self, batch_size, is_train=True, global_step=None, store_ll=True):
        """Samples the required number of high-level transitions. Number of LL transitions can be higher."""
        na = self._hp.number_of_agents

        hl_experience_batch, ll_experience_batch = [], []
        env_steps, hl_step = 0, 0
        with self._env.val_mode() if not is_train else contextlib.suppress():
            with self._agent.val_mode() if not is_train else contextlib.suppress():
                with self._agent.rollout_mode():
                    while hl_step < batch_size or len(ll_experience_batch) <= 1:
                        # perform one rollout step
                        # agent_output = self.sample_action(self._obs)
                        # agent_output = self._postprocess_agent_output(agent_output)

                        agent_output = [self.sample_action(self._obs[agent_index]) for agent_index in range(na)]
                        actions = [agent_output[agent_index].action for agent_index in range(na)]

                        obs, reward, done, info = self._env.step(actions)
                        # obs = self._postprocess_obs(obs)

                        # update last step's 'observation_next' with HL action
                        if store_ll:
                            if ll_experience_batch:
                                ll_experience_batch[-1].observation_next = \
                                    self._agent.make_ll_obs(ll_experience_batch[-1].observation_next, agent_output.hl_action)

                            # store current step in ll_experience_batch
                            ll_experience_batch.append(AttrDict(
                                observation=self._agent.make_ll_obs(self._obs, agent_output.hl_action),
                                reward=reward,
                                done=done,
                                action=agent_output.action,
                                observation_next=obs,       # this will get updated in the next step
                            ))

                        # store HL experience batch if this was HL action or episode is done
                        if agent_output.is_hl_step or (done or self._episode_step >= self._max_episode_len-1):
                            if self.last_hl_obs is not None and self.last_hl_action is not None:
                                hl_experience_batch.append(AttrDict(
                                    observation=self.last_hl_obs,
                                    reward=self.reward_since_last_hl,
                                    done=done,
                                    action=self.last_hl_action,
                                    observation_next=obs,
                                ))
                                hl_step += 1
                                if done:
                                    hl_experience_batch[-1].reward += reward  # add terminal reward
                                if hl_step % 1000 == 0:
                                    print("Sample step {}".format(hl_step))
                            self.last_hl_obs = self._obs if self._episode_step == 0 else obs
                            self.last_hl_action = agent_output.hl_action
                            self.reward_since_last_hl = 0

                        # update stored observation
                        self._obs = obs
                        env_steps += 1; self._episode_step += 1; self._episode_reward += reward
                        self.reward_since_last_hl += reward

                        # reset if episode ends
                        if done or self._episode_step >= self._max_episode_len:
                            if not done:    # force done to be True for timeout
                                ll_experience_batch[-1].done = True
                                if hl_experience_batch:   # can potentially be empty 
                                    hl_experience_batch[-1].done = True
                            self._episode_reset(global_step)
        return AttrDict(
            hl_batch=listdict2dictlist(hl_experience_batch),
            ll_batch=listdict2dictlist(ll_experience_batch[:-1]),   # last element does not have updated obs_next!
        ), env_steps


    def _episode_reset(self, global_step=None):
        super()._episode_reset(global_step)
        self.last_hl_obs, self.last_hl_action = None, None
        self.reward_since_last_hl = 0