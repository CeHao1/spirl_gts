import os
import torch
import numpy as np
from spirl.utils.pytorch_utils import map2torch, map2np

from spirl.rl.train import RLTrainer
from spirl.rl.components.params import get_args
from spirl.train import  set_seeds, make_path
from spirl.utils.general_utils import AttrDict, ParamDict, AverageTimer, timing, pretty_print
from spirl.rl.utils.rollout_utils import RolloutSaver
from spirl.rl.utils.mpi import update_with_mpi_config, set_shutdown_hooks

from spirl.utils.gts_utils import load_standard_table
import matplotlib.pyplot as plt


class HRLVisualizer(RLTrainer):
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

        self.agent = self._hp.agent(self.conf.agent)
        self.agent.to(self.device)

        # if self.conf.ckpt_path is not None:
        start_epoch = self.resume(args.resume, self.conf.ckpt_path)

        self.state_scaler, self.action_scaler = load_standard_table()

        # this is real observation, and action(z)
        # dict_keys(['action', 'reward', 'done', 'observation', 'observation_next'])

        # replay_buffer = self.agent.hl_agent.replay_buffer
        # one_sample = replay_buffer.sample(1)
        # self.decode_hl_actions(one_sample)


        # this is real action(steering and pedal)
        # dict_keys(['actions', 'done', 'pad_mask', 'reward', 'states'])
        saver = RolloutSaver('./sample/hrl/no_prior/')
        inputs = saver.load_rollout_to_file(0)
        self.test_policy_and_prior(inputs)


    def test_policy_and_prior(self, inputs):
        obs = inputs['states']
        obs = torch.from_numpy(obs).to(self.device)
        
        # from obs to hl actions z 
        hl_policy_musig = self.agent.hl_agent.policy.net(obs).detach().cpu().numpy()
        hl_output = self.agent.hl_agent.act(obs)
        # print(hl_output)

        hl_action = hl_output['action']
        # from obs, z to 
        ll_actions = self.decode_hl_actions(obs, hl_action)
        ll_actions = self.action_scaler.inverse_transform(ll_actions)
        for idx in np.random.choice(ll_actions.shape[0], 20, False):
            ll_action = ll_actions[idx]
            self.plot_action_series(ll_action)
        

    def decode_hl_actions(self, obs, hl_action):
        # obs = one_sample['observation']
        # hl_action = one_sample['action']
        if (isinstance(obs, np.ndarray)):
            obs = torch.from_numpy(obs).to(self.device)
        hl_action = torch.from_numpy(hl_action).to(self.device)
        
        # print('obs', obs)
        # print('hl action', hl_action)

        output = self.agent.ll_agent._policy.decode(hl_action, hl_action, self.agent.ll_agent._policy.n_rollout_steps)
        ll_actions = map2np(output)
        return ll_actions

        # print('ll_actions', ll_actions)

    def plot_action_series(self, action):
        
        plt.figure(figsize=(12,4))
        plt.subplot(1,2,1)
        plt.plot(action[:,0], label='steering')

        plt.subplot(1,2,2)
        plt.plot(action[:,1], label='pedal')

        plt.legend()
        plt.show()




if __name__ == '__main__':
    HRLVisualizer(args=get_args())
