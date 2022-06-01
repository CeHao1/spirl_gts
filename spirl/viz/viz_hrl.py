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

from spirl.utils.gts_utils import load_standard_table, obs2name
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
        print('inputs', inputs.states.shape)
        self.test_policy_and_prior(inputs)
        # self.plot_actions(inputs)

        


    def test_policy_and_prior(self, inputs):
        obs = inputs['states']
        # obs = torch.from_numpy(obs).to(self.device)
        
        # from obs to hl actions z 
        # hl_policy_musig = self.agent.hl_agent.policy.net(obs).detach().cpu().numpy()

        idx = np.random.choice(obs.shape[0], 20, False)
        obs = obs[idx, :]
        act = inputs['actions'][idx, :]
        obs_tensor = torch.from_numpy(obs).to(self.device)

        hl_output = self.agent.hl_agent.act(obs_tensor)
        hl_action = hl_output['action']
        # ll_actions = self.decode_hl_actions(obs_tensor, hl_action)
        hl_action = torch.from_numpy(hl_action).to(self.device)
        output = self.agent.ll_agent._policy.decode(hl_action, obs, self.agent.ll_agent._policy.n_rollout_steps)
        ll_actions = map2np(output)

        ll_actions = self.action_scaler.inverse_transform(ll_actions)
        act = self.action_scaler.inverse_transform(act)

        # inputs2 = AttrDict()
        # inputs2.states = obs
        # print(obs.shape)
        # prior_output = self.agent.ll_agent._policy.run(inputs, use_learned_prior=True, output_actions=False)

        # print('act', act)

        # obs = map2np(obs)
        # obs = self.state_scaler.inverse_transform(obs)

        for ll_action, one_obs, one_act in zip(ll_actions, obs, act):
            no_pop_output = self.agent.no_pop_act(one_obs)
            no_pop_act = no_pop_output.action
            no_pop_act = self.action_scaler.inverse_transform(no_pop_act)

            self.agent._steps_since_hl = 0
            self.agent.ll_agent.reset()
            agent_output = self.agent.act(one_obs)
            agent_act = self.action_scaler.inverse_transform([agent_output.action])[0]

            # inputs2 = AttrDict()
            # inputs2.states = one_obs
            # prior_output = self.agent.ll_agent._policy.run(inputs2, use_learned_prior=True, output_actions=False)

            obs_for_state = self.state_scaler.inverse_transform([one_obs])[0]
            state = obs2name(obs_for_state)
            self.plot_action_series(ll_action, no_pop_act,  state, one_act, agent_act)
              

    def decode_hl_actions(self, obs, hl_action):
        # obs = one_sample['observation']
        # hl_action = one_sample['action']
        if (isinstance(obs, np.ndarray)):
            obs = torch.from_numpy(obs).to(self.device)
        hl_action = torch.from_numpy(hl_action).to(self.device)
        
        # output = self.agent.ll_agent._policy.decode(hl_action, hl_action, self.agent.ll_agent._policy.n_rollout_steps)
        output = self.agent.ll_agent._policy.decode(hl_action, obs, self.agent.ll_agent._policy.n_rollout_steps)
        ll_actions = map2np(output)
        return ll_actions


    def plot_action_series(self, action, no_pop_act, state, act, agent_act):
        rad2deg = 180.0 / np.pi
        range2deg = 180.0 / 6.0

        print(state)
        
        plt.figure(figsize=(12,4))
        plt.subplot(1,2,1)
        plt.plot(action[:,0] * range2deg, 'b', label='ll action, manual')
        plt.plot(no_pop_act[:,0] * range2deg, 'g', label='no_pop_act')
        plt.plot( state['delta'] * rad2deg, 'bo', label='last delta' )
        plt.plot(act[0] * range2deg, 'ro', label='real delta' )
        plt.plot(agent_act[0] * range2deg, 'go', label='new agent delta' )
        plt.title('steering')
        plt.legend()

        plt.subplot(1,2,2)
        plt.plot(action[:,1], 'b', label='ll action, manual')
        plt.plot(no_pop_act[:,1], 'g', label='no_pop_act')
        plt.plot(state['thr']-state['brk'], 'bo', label='last pedal')
        plt.plot(act[1], 'ro', label='real pedal')
        plt.plot(agent_act[1], 'ro', label='new agent pedal')
        plt.ylim([min(-1, min(action[:,1])-0.1), max(1, max(action[:,1])+0.1)])
        plt.title('pedal')
        plt.legend()
        plt.show()

        # 
        plt.figure(figsize=(12,4))

        x = np.arange(10)
        I = np.ones(10)
        plt.subplot(1,2,1)
        plt.plot([0], state['ey'], 'ro', label='ey')
        plt.plot(x, I * (state['Wl'] + state['ey']))
        plt.plot(x, -I * (state['Wr'] - state['ey']))
        plt.title('lateral dist')
        plt.legend()

        plt.subplot(1,2,2)
        plt.plot(state['kap'], 'b--', label='kap')
        plt.title('curvature')
        plt.legend()

        plt.show()

    def plot_actions(self, inputs):
        act = inputs.actions
        plt.figure(figsize=(15,10))
        plt.subplot(2,1,1)
        plt.plot(act[:,0], 'b.')
        plt.title('real delta')

        plt.subplot(2,1,2)
        plt.plot(act[:,1], 'b.')
        plt.title('real pedal')
        plt.show()



if __name__ == '__main__':
    HRLVisualizer(args=get_args())
