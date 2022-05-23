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
        saver = RolloutSaver(self.args.save_dir)
        sampled_data = saver.load_roolout_to_file(0)
        print('set up the viz')
        self.replay2actions(sampled_data)

    def replay2actions(self, sampled_data):
        num_of_samples = 1000
        # sampled_data = self.agent.replay_buffer.sample(num_of_samples, random=False)

        obs = sampled_data.states
        rew = sampled_data.reward
        act = sampled_data.actions
        done = sampled_data.done
        done_at = np.where(done == True)[0]


        obs_t = torch.from_numpy(obs).to(self.device)
        output = self.agent.policy.net(obs_t).detach().cpu().numpy()

        titles = ['steering mean', 'pedal mean', 'steering std', 'pedal std']



        plt.figure(figsize=(15,10))
        for i in range(4):
            plt.subplot(2,2, i+1)

            data = output[:, i]
            if i >= 2:
                data = np.exp(data)
            plt.plot(data, 'b.')
            plt.title(titles[i])

        plt.show()




        # print(output)


if __name__ == '__main__':
    RLVisualizer(args=get_args())

