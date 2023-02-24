import os
import torch
import numpy as np
from spirl.rl.train import RLTrainer
from spirl.rl.components.params import get_args
from spirl.train import  set_seeds, make_path
from spirl.utils.general_utils import AttrDict, ParamDict, AverageTimer, timing, pretty_print
from spirl.rl.utils.rollout_utils import RolloutSaver
from spirl.rl.utils.mpi import update_with_mpi_config, set_shutdown_hooks

import matplotlib.pyplot as plt

class RLVisualizer(RLTrainer):

    def __init__(self, args):
        self.args = args
        self.setup_device()

        # set up params
        self.conf = self.get_config()
        update_with_mpi_config(self.conf)   # self.conf.mpi = AttrDict(is_chef=True)
        self._hp = self._default_hparams()

        self._hp.overwrite(self.conf.general)  # override defaults with config file
        self._hp.exp_path = make_path(self.conf.exp_dir, args.path, args.prefix, args.new_dir)
        self.log_dir = log_dir = os.path.join(self._hp.exp_path, 'log')
        print('using log dir: ', log_dir)

        # set seeds, display, worker shutdown
        if args.seed != -1: self._hp.seed = args.seed   # override from command line if set
        set_seeds(self._hp.seed)
        os.environ["DISPLAY"] = ":1"
        set_shutdown_hooks()

        # set up logging
        # if self.is_chef:
        #     print("Running base worker.")
        #     self.logger = self.setup_logging(self.conf, self.log_dir)
        # else:
        #     print("Running worker {}, disabled logging.".format(self.conf.mpi.rank))
        #     self.logger = None

        # build env
        # self.conf.env.seed = self._hp.seed
        # if 'task_params' in self.conf.env: self.conf.env.task_params.seed=self._hp.seed
        # if 'general' in self.conf: self.conf.general.seed=self._hp.seed
        # self.env = self._hp.environment(self.conf.env)
        # self.conf.agent.env_params = self.env.agent_params      # (optional) set params from env for agent
        # if self.is_chef:
        #     pretty_print(self.conf)

        # build agent (that holds actor, critic, exposes update method)
        # self.conf.agent.num_workers = self.conf.mpi.num_workers
        self.agent = self._hp.agent(self.conf.agent)
        self.agent.to(self.device)

        # if self.conf.ckpt_path is not None:
        start_epoch = self.resume(args.resume, self.conf.ckpt_path)
        saver = RolloutSaver('./sample/rl/sac')
        sampled_data = saver.load_rollout_to_file(0)
        print('set up the viz')
        self.replay2actions(sampled_data)
        # self.replay2Q(sampled_data)

        # print('policy', self.agent.policy.net)
        # print('critics', self.agent.critics)

        # self.show_replay_buffer()


    def replay2actions(self, sampled_data):
        num_of_samples = 1000
        # sampled_data = self.agent.replay_buffer.sample(num_of_samples, random=False)
        print('keys', sampled_data.keys())
    
        obs = sampled_data.states[-num_of_samples:-1]
        rew = sampled_data.reward[-num_of_samples:-1]
        act = sampled_data.actions[-num_of_samples:-1]
        done = sampled_data.done[-num_of_samples:-1]
        done_at = np.where(done == True)[0]

        # print('rew', rew)

        obs_t = torch.from_numpy(obs).to(self.device)
        output = self.agent.policy.net(obs_t).detach().cpu().numpy()

        titles = ['steering mean', 'pedal mean', 'steering std', 'pedal std']


        # mean
        plt.figure(figsize=(17,5))
        plt.subplot(1,2, 1)
        plt.plot(output[:, 0], 'b.')
        plt.title(titles[0], fontsize=20)
        plt.subplot(1, 2, 2)
        plt.plot(output[:, 1], 'b.')
        plt.title(titles[1], fontsize=20)
        plt.show()

        # std
        plt.figure(figsize=(17,5))
        plt.subplot(1,2, 1)
        plt.plot(np.exp(output[:, 2]), 'b.')
        plt.title(titles[2], fontsize=20)
        plt.subplot(1, 2, 2)
        plt.plot(np.exp(output[:, 3]), 'b.')
        plt.title(titles[3], fontsize=20)
        plt.show()

        # action
        plt.figure(figsize=(17,5))
        plt.subplot(1,2, 1)
        plt.plot(act[:, 0], 'b.')
        plt.title('steer action', fontsize=20)
        plt.subplot(1, 2, 2)
        plt.plot(act[:, 1], 'b.')
        plt.title('pedal action', fontsize=20)
        plt.show()

        # reward
        plt.figure(figsize=(10,5))
        plt.plot(rew, 'b.')
        plt.title('rewards')
        plt.show()


    def replay2Q(self, sampled_data):
        num_of_samples = 1000
        experience_batch = AttrDict()
        experience_batch.observation = torch.from_numpy(sampled_data.states[-num_of_samples:-1]).to(self.device)
        experience_batch.action = torch.from_numpy(sampled_data.actions[-num_of_samples:-1]).to(self.device)

        qs = self.agent._compute_q_estimates(experience_batch)
        q1 = qs[0].detach().cpu().numpy()
        q2 = qs[1].detach().cpu().numpy()

        # policy_output_next = self._run_policy(experience_batch.observation_next)
        # value_next = self._compute_next_value(experience_batch, policy_output_next)
        # q_target = experience_batch.reward * self._hp.reward_scale + \
        #                 (1 - experience_batch.done) * self._hp.discount_factor * value_next
        # if self._hp.clip_q_target:
        #     q_target = self._clip_q_target(q_target)
        # q_target = q_target.detach()


        plt.figure(figsize=(17,5))
        plt.plot(q1, 'b.')
        plt.title('q1')
        plt.show()

        plt.figure(figsize=(17,5))
        plt.plot(q2, 'b.')
        plt.title('q2')
        plt.show()


        # print('qs !!', qs)


    def show_replay_buffer(self):
        sampled_batch = self.agent.replay_buffer.sample(2)
        print(sampled_batch)

if __name__ == '__main__':
    RLVisualizer(args=get_args())

