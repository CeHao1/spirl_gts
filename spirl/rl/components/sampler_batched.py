import numpy as np
import contextlib
from collections import deque

from spirl.utils.general_utils import listdict2dictlist, batch_listdict2dictlist, AttrDict, ParamDict, obj2np
from spirl.modules.variational_inference import MultivariateGaussian
from spirl.rl.utils.reward_fcns import sparse_threshold

class SamplerBatched:
    """Collects rollouts from the environment using the given agent."""
    def __init__(self, config, env, agent, logger, max_episode_len):
        self._hp = self._default_hparams().overwrite(config)

        self._env = env
        self._agent = agent
        self._logger = logger
        self._max_episode_len = max_episode_len

        self._obs = None
        self._episode_step, self._episode_reward = 0, 0

    def _default_hparams(self):
        return ParamDict({
            'sample_complete_rollout_length' : True,
            'select_agent_id': [],
        })

    def init(self, is_train):
        """Starts a new rollout. Render indicates whether output should contain image."""
        with self._env.val_mode() if not is_train else contextlib.suppress():
            with self._agent.val_mode() if not is_train else contextlib.suppress():
                self._episode_reset()

    def sample_action(self, obs):
        return self._agent.act(obs)

    def sample_batch(self, batch_size, is_train=True, global_step=None):
        """Samples an experience batch of the required size."""
        experience_batch = []
        step = 0
        with self._env.val_mode() if not is_train else contextlib.suppress():
            with self._agent.val_mode() if not is_train else contextlib.suppress():
                with self._agent.rollout_mode():
                    # reset again for gts
                    self._episode_reset(global_step)
                    # while step < batch_size or (self._episode_step != 0): # must complete one episode
                    while step < batch_size:
                        # perform one rollout step
                        agent_output = self.sample_action(self._obs)
                        if agent_output.action is None:
                            self._episode_reset(global_step)
                            continue
                        agent_output = self._postprocess_agent_output(agent_output)

                        obs, reward, done, info = self._env.step(agent_output.action)
                        obs, reward, done, info = self._select_agent_id(obs, reward, done, info)
                        
                        obs = self._postprocess_obs(obs)
                        assert len(obs.shape) == 2 # must be batched array. first dim is batch size, second dim is the obs data
                        batch_length = obs.shape[0]
                                        
                        experience_batch.append(AttrDict(
                            observation=self._obs,
                            reward=reward,
                            done=done,
                            action=agent_output.action,
                            observation_next=obs,
                            info=info,
                        ))

                        # update stored observation
                        self._obs = obs
                        step += batch_length; 
                        self._episode_step += 1; 
                        self._episode_reward += np.mean(reward)

                        # reset if episode ends
                        if np.any(done) or self._episode_step >= self._max_episode_len:
                            if np.any(done):
                                print('restart, done one sample')
                            if self._episode_step >= self._max_episode_len:
                                print(f'restart, step {self._episode_step} exceed max episode len {self._max_episode_len}')
                            if not np.all(done):    # force done to be True for timeout
                                for exp in experience_batch[-1].done:
                                    exp = True
                            self._episode_reset(global_step)

        return batch_listdict2dictlist(experience_batch), step

    def sample_episode(self, is_train, render=False, deterministic_action=False):
        """Samples one episode from the environment."""
        self.init(is_train)
        episode, done = [], False
        with self._env.val_mode() if not is_train else contextlib.suppress():
            with self._agent.val_mode() if not is_train else contextlib.suppress():
                with self._agent.rollout_mode():
                    while not np.any(done):
                        # perform one rollout step
                        agent_output = self.sample_action(self._obs)
                        agent_output = self._postprocess_agent_output(agent_output, deterministic_action=deterministic_action)
                        if render:
                            render_obs = self._env.render()

                        # print(agent_output.action)
                        obs, reward, done, info = self._env.step(agent_output.action)
                        obs, reward, done, info = self._select_agent_id(obs, reward, done, info)
                        assert len(obs.shape) == 2
                        # batch_length = obs.shape[0]

                        obs = self._postprocess_obs(obs)
                        episode.append(AttrDict(
                            observation=self._obs,
                            reward=reward,
                            done=done,
                            action=agent_output.action,
                            observation_next=obs,
                            info=obj2np(info),
                        ))
                        if render:
                            episode[-1].update(AttrDict(image=render_obs))

                        
                        # update stored observation
                        self._obs = obs
                        self._episode_step += 1
                        self._episode_reward += np.mean(reward)

        for exp in episode[-1].done:# make sure episode is marked as done at final time step
            exp = True

        return batch_listdict2dictlist(episode)

    def get_episode_info(self):
        episode_info = AttrDict(episode_reward=self._episode_reward,
                                episode_length=self._episode_step,
                                episode_success=np.clip(self._episode_reward, 0, 1))
        if hasattr(self._env, "get_episode_info"):
            episode_info.update(self._env.get_episode_info())
        return episode_info

    def _episode_reset(self, global_step=None):
        """Resets sampler at the end of an episode."""
        self._log_episode_info(global_step)
        self._episode_step, self._episode_reward = 0, 0.
        self._obs = self._postprocess_obs(self._select_one_agent_id(self._reset_env()))
        self._agent.reset()
        
    def _log_episode_info(self, global_step):
        """Logs episode info."""
        if self._logger is not None and global_step is not None:
            self._logger.log_scalar_dict(self.get_episode_info(),
                                         prefix='train' if self._agent._is_train else 'val',
                                         step=global_step)

    def _reset_env(self):
        return self._env.reset()

    def _select_one_agent_id(self, obs):
        if self._hp.select_agent_id:
            obs = obs[self._hp.select_agent_id]
        return obs

    def _select_agent_id(self, obs, reward, done, info):
        if self._hp.select_agent_id:
            obs = obs[self._hp.select_agent_id]
            reward = reward[self._hp.select_agent_id]
            done = done[self._hp.select_agent_id]
            info = info[self._hp.select_agent_id]
        return obs, reward, done, info

    def _postprocess_obs(self, obs):
        """Optionally post-process observation."""
        return obs

    def _postprocess_agent_output(self, agent_output, deterministic_action=False):
        """Optionally post-process / store agent output."""
        if deterministic_action:
            if isinstance(agent_output.dist, MultivariateGaussian):
                agent_output.ori_action = agent_output.action
                agent_output.action = agent_output.dist.mean
        return agent_output


class HierarchicalSamplerBatched(SamplerBatched):
    """Collects experience batches by rolling out a hierarchical agent. Aggregates low-level batches into HL batch."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_hl_obs, self.last_hl_action = None, None  # stores observation when last hl action was taken
        self.reward_since_last_hl = 0  # accumulates the reward since the last HL step for HL transition

    def sample_batch(self, batch_size, is_train=True, global_step=None, store_ll=True):
        """Samples the required number of high-level transitions. Number of LL transitions can be higher."""
        hl_experience_batch, ll_experience_batch = [], []
        env_steps, hl_step = 0, 0
        with self._env.val_mode() if not is_train else contextlib.suppress():
            with self._agent.val_mode() if not is_train else contextlib.suppress():
                with self._agent.rollout_mode():
                    # reset again for gts
                    self._episode_reset(global_step)
                    
                    # while env_steps < batch_size or (self._episode_step != 0): #must complete one sampling
                    while env_steps < batch_size:
                        # perform one rollout step
                        agent_output = self.sample_action(self._obs)
                        agent_output = self._postprocess_agent_output(agent_output)
                        obs, reward, done, info = self._env.step(agent_output.action)
                        obs, reward, done, info = self._select_agent_id(obs, reward, done, info)
                        assert len(obs.shape) == 2
                        batch_length = obs.shape[0]

                        obs = self._postprocess_obs(obs)

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
                                info=info,
                            ))

                        # store HL experience batch if this was HL action or episode is done
                        if agent_output.is_hl_step or (np.any(done) or self._episode_step >= self._max_episode_len-1):
                            if self.last_hl_obs is not None and self.last_hl_action is not None:
                                hl_experience_batch.append(AttrDict(
                                    observation=self.last_hl_obs,
                                    reward=self.reward_since_last_hl,
                                    done=done,
                                    action=self.last_hl_action,
                                    observation_next=obs,
                                    info=info,
                                ))
                                hl_step += batch_length
                                if np.any(done):
                                    # add terminal reward
                                    for exp, r in zip(hl_experience_batch[-1].reward, reward):
                                        exp += r
                                # if hl_step % 1000 == 0:
                                #     print("Sample step {}".format(hl_step))
                            self.last_hl_obs = self._obs if self._episode_step == 0 else obs
                            self.last_hl_action = agent_output.hl_action
                            self.reward_since_last_hl = 0

                        # update stored observation
                        self._obs = obs
                        env_steps += batch_length; 
                        self._episode_step += 1; 
                        self._episode_reward += np.mean(reward)
                        self.reward_since_last_hl += np.array(reward)

                        # reset if episode ends
                        if np.any(done) or self._episode_step >= self._max_episode_len:
                            if not np.all(done):    # force done to be True for timeout
                                for exp in ll_experience_batch[-1].done:
                                    exp = True
                                if hl_experience_batch:   # can potentially be empty 
                                    for exp in hl_experience_batch[-1].done:
                                        exp = True
                            print('!! done reset, _episode_step: {}, hl_step: {}'.format(self._episode_step, hl_step))
                            self._episode_reset(global_step)
        return AttrDict(
            hl_batch=batch_listdict2dictlist(hl_experience_batch),
            ll_batch=batch_listdict2dictlist(ll_experience_batch[:-1]),   # last element does not have updated obs_next!
        ), env_steps

    def _episode_reset(self, global_step=None):
        super()._episode_reset(global_step)
        self.last_hl_obs, self.last_hl_action = None, None
        self.reward_since_last_hl = 0


class AgentDetached_SampleBatched(SamplerBatched):
    
    def sample_batch(self, batch_size, is_train=True, global_step=None):
        self._init_batch_episode_info()
        experience_batch, env_step = super().sample_batch(batch_size, is_train, global_step)
        episode_info = self._summary_batch_episode_info(global_step)
        return [experience_batch, env_step, episode_info]
    
    def _log_episode_info(self, global_step):
        if global_step is not None:
            self._append_batch_episode_info(self.get_episode_info())
    
    def _init_batch_episode_info(self):
        self._batch_episode_info = []
        
    def _append_batch_episode_info(self, episode_info):
        self._batch_episode_info.append(episode_info)
        
    def _summary_batch_episode_info(self, global_step=None):
        episode_info = AttrDict()
        if global_step is not None and self._logger is not None:
            for key in self._batch_episode_info[0].keys():
                episode_info[key] = np.mean([info[key] for info in self._batch_episode_info])
        return episode_info

class AgentDetached_HierarchicalSamplerBatched(HierarchicalSamplerBatched):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_hl_output = None

    def sample_action(self, obs):
        """Samples an action from the agent."""
        agent_output = self._agent.act(obs, self._last_hl_output)
        if agent_output.is_hl_step:
            self._last_hl_output = AttrDict(
                action=agent_output.hl_action,
                dist=agent_output.hl_dist,
                log_prob=agent_output.hl_log_prob,
            )
        return agent_output
    
    def sample_batch(self, batch_size, is_train=True, global_step=None):
        self._init_batch_episode_info()
        experience_batch, env_step = super().sample_batch(batch_size, is_train, global_step)
        episode_info = self._summary_batch_episode_info(global_step)
        return [experience_batch, env_step, episode_info]
        
    def _log_episode_info(self, global_step):
        AgentDetached_SampleBatched._log_episode_info(self, global_step)

    def _init_batch_episode_info(self):
        AgentDetached_SampleBatched._init_batch_episode_info(self)

    def _append_batch_episode_info(self, episode_info):
        AgentDetached_SampleBatched._append_batch_episode_info(self, episode_info)

    def _summary_batch_episode_info(self, global_step=None):
        return AgentDetached_SampleBatched._summary_batch_episode_info(self, global_step)
    
    def _episode_reset(self, global_step=None):
        super()._episode_reset(global_step)
        self._last_hl_output = None

    