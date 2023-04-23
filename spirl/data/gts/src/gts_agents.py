
from spirl.rl.agents.ac_agent import SACAgent
from spirl.rl.agents.prior_sac_agent import ActionPriorSACAgent
from spirl.rl.agents.skill_critic.hl_inherit_agent import HLInheritAgent
from spirl.rl.agents.skill_critic.ll_inherit_agent import LLInheritAgent

from spirl.rl.envs.gts_corner2.gts_corner2_single import GTSEnv_Corner2_Single

from spirl.utils.general_utils import AttrDict
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class GTSAgent:

    VIS_RANGE = GTSEnv_Corner2_Single.VIS_RANGE
    START_POS = GTSEnv_Corner2_Single.START_POS
    TARGET_POS = GTSEnv_Corner2_Single.TARGET_POS
    KEY_LIST = ['pos[0]', 'pos[2]', 'vx', 'steering','throttle', 'brake', 'is_hit_wall']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.info_replay_buffer = self._hp.replay(self._hp.replay_params)

    def visualize(self, logger, rollout_storage, step):
        self._vis_replay_buffer(logger, step)

    def _vis_replay_buffer(self, logger, step):
        """Visualizes a replay buffer."""
        size = self.info_replay_buffer.size
        states = self.info_replay_buffer.get().observation[:size, :]
        plot_gts_traj(states, logger, step, size)

    def add_experience(self, experience_batch):
        if 'info' in experience_batch:
            self.info_replay_buffer.append(self._select_info(experience_batch.pop('info')))
        super().add_experience(experience_batch)

    def update(self, experience_batch=None):
        if 'info' in experience_batch:
            self.info_replay_buffer.append(self._select_info(experience_batch.pop('info')))
        return super().update(experience_batch)

    def _select_info(self, info):
        """Selects info to be stored in replay buffer."""
        key_list = GTSAgent.KEY_LIST
        info_batch = AttrDict(
            observation = np.array([[info_t['state'][key] for key in key_list] for info_t in info])
        )
        return info_batch


class GTSSACAgent(GTSAgent, SACAgent):
    def visualize(self, logger, rollout_storage, step):
        GTSAgent.visualize(self, logger, rollout_storage, step)
        SACAgent.visualize(self, logger, rollout_storage, step)
    
class GTSActionPriorSACAgent(ActionPriorSACAgent, GTSAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.info_replay_buffer = self._hp.replay(self._hp.replay_params)

    def visualize(self, logger, rollout_storage, step):
        GTSAgent.visualize(self, logger, rollout_storage, step)
        self._vis_hl_q(logger, step)

    def add_experience(self, experience_batch):
        self.info_replay_buffer.append(self._select_info(experience_batch.pop('info')))
        super().add_experience(experience_batch)

    def _vis_hl_q(self, logger, step):
        """Visualizes high-level Q function."""
        self._vis_q(logger, step, prefix='hl', plot_type='gts')

class GTSHLInheritAgent(HLInheritAgent, GTSAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.info_replay_buffer = self._hp.replay(self._hp.replay_params)

    def visualize(self, logger, rollout_storage, step):
        GTSAgent.visualize(self, logger, rollout_storage, step)
        self._vis_hl_q(logger, step)

    def add_experience(self, experience_batch):
        self.info_replay_buffer.append(self._select_info(experience_batch.pop('info')))
        super().add_experience(experience_batch)

    def _vis_hl_q(self, logger, step):
        """Visualizes high-level Q function."""
        size = self.info_replay_buffer.size
        states = self.info_replay_buffer.get().observation[:size, :]
        self._vis_q(logger, step, prefix='hl', plot_type='gts', external_states=states)

class GTSLLInheritAgent(LLInheritAgent, GTSAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.info_replay_buffer = self._hp.replay(self._hp.replay_params)

    def visualize(self, logger, rollout_storage, step):
        GTSAgent.visualize(self, logger, rollout_storage, step)
        self._vis_ll_q(logger, step)

    def add_experience(self, experience_batch):
        self.info_replay_buffer.append(self._select_info(experience_batch.pop('info')))
        super().add_experience(experience_batch)

    def _vis_ll_q(self, logger, step):
        """Visualizes high-level Q function."""
        size = self.info_replay_buffer.size
        states = self.info_replay_buffer.get().observation[:size, :]
        self._vis_q(logger, step, prefix='ll', plot_type='gts', external_states=states,
                    content=['q', 'KLD', 'action', 'action_nosquash', 'action_recent', 'action_nosquash_recent'])
    

def plot_gts_traj(states, logger, step, size):

    sdict = {}
    for i in range(len(GTSAgent.KEY_LIST)):
        sdict[GTSAgent.KEY_LIST[i]] = states[:, i]

    sdict['steering'] /= np.pi/6

    # plot replay with velocity
    fig = plt.figure(figsize=(14,8))
    plt.scatter(sdict['pos[0]'], sdict['pos[2]'], c=sdict['vx'], cmap='Reds', s=0.1)
    plt.plot(GTSAgent.START_POS[0], GTSAgent.START_POS[1], 'go')
    plt.plot(GTSAgent.TARGET_POS[0], GTSAgent.TARGET_POS[1], 'mo')
    plt.axis("equal")
    plt.title('velocity, step ' + str(step) + ' size ' + str(size))
    # plt.xlim(GTSAgent.VIS_RANGE[0])
    # plt.ylim(GTSAgent.VIS_RANGE[1])
    plt.colorbar()
    logger.log_plot(fig, "replay, velocity_vis", step)
    plt.close(fig)


    # plot replay with density
    fig = plt.figure(figsize=(14,8))
    sns.histplot(sdict['pos[0]'], sdict['pos[2]'], cmap='Blues', cbar=True,)
    plt.plot(GTSAgent.START_POS[0], GTSAgent.START_POS[1], 'go')
    plt.plot(GTSAgent.TARGET_POS[0], GTSAgent.TARGET_POS[1], 'mo')
    plt.axis("equal")
    plt.title('density, step ' + str(step) + ' size ' + str(size))
    # plt.xlim(GTSAgent.VIS_RANGE[0])
    # plt.ylim(GTSAgent.VIS_RANGE[1])
    logger.log_plot(fig, "replay, density_vis", step)
    plt.close(fig)

    # plot replay hit wall
    fig = plt.figure(figsize=(14,8))
    plt.scatter(sdict['pos[0]'], sdict['pos[2]'], c=sdict['is_hit_wall'], cmap='Reds', s=0.1)
    plt.plot(GTSAgent.START_POS[0], GTSAgent.START_POS[1], 'go')
    plt.plot(GTSAgent.TARGET_POS[0], GTSAgent.TARGET_POS[1], 'mo')
    plt.axis("equal")
    plt.title('hit wall, step ' + str(step) + ' size ' + str(size))
    # plt.xlim(GTSAgent.VIS_RANGE[0])
    # plt.ylim(GTSAgent.VIS_RANGE[1])
    plt.colorbar()
    logger.log_plot(fig, "replay, hit_wall_vis", step)
    plt.close(fig)

    # plot steering
    fig = plt.figure(figsize=(14,8))
    plt.scatter(sdict['pos[0]'], sdict['pos[2]'], c=sdict['steering'], cmap='jet', s=0.1)
    plt.plot(GTSAgent.START_POS[0], GTSAgent.START_POS[1], 'go')
    plt.plot(GTSAgent.TARGET_POS[0], GTSAgent.TARGET_POS[1], 'mo')
    plt.axis("equal")
    plt.title('steering Blue(-right), Red(+left), step ' + str(step) + ' size ' + str(size))
    # plt.xlim(GTSAgent.VIS_RANGE[0])
    # plt.ylim(GTSAgent.VIS_RANGE[1])
    plt.colorbar()
    logger.log_plot(fig, "replay, steering_vis", step)
    plt.close(fig)

    # plot pedal
    fig = plt.figure(figsize=(14,8))
    plt.scatter(sdict['pos[0]'], sdict['pos[2]'], c=sdict['throttle']-sdict['brake'], cmap='jet', s=0.1)
    plt.plot(GTSAgent.START_POS[0], GTSAgent.START_POS[1], 'go')
    plt.plot(GTSAgent.TARGET_POS[0], GTSAgent.TARGET_POS[1], 'mo')
    plt.axis("equal")
    plt.title('Pedal Blue(-dec), Red(+acc), step ' + str(step) + ' size ' + str(size))
    # plt.xlim(GTSAgent.VIS_RANGE[0])
    # plt.ylim(GTSAgent.VIS_RANGE[1])
    plt.colorbar()
    logger.log_plot(fig, "replay, pedal_vis", step)
    plt.close(fig)

    # plot action dist of replay
    fs = 16
    fig = plt.figure(figsize=(10, 8))
    sns.histplot(x=sdict['throttle'], y=sdict['brake'], bins=50, cmap='Reds', cbar=True)
    plt.xlabel('dim_0', fontsize=fs)
    plt.ylabel('dim_1', fontsize=fs)
    plt.title('2D dist of latent variables, size ' + str(size), fontsize=fs)
    logger.log_plot(fig, name= 'replay action dist', step=step)
    plt.close(fig)



def plot_gts_value(q, states, logger, step, size, fig_name='vis'):
    fig = plt.figure(figsize=(14,8))
    plt.scatter(states[:, 0], states[:, 1], s=0.1, c=q, cmap='Oranges')
    plt.plot(GTSAgent.START_POS[0], GTSAgent.START_POS[1], 'go')
    plt.plot(GTSAgent.TARGET_POS[0], GTSAgent.TARGET_POS[1], 'mo')
    plt.axis("equal")
    plt.title(fig_name + ' step ' + str(step) + ' size ' + str(size))
    # plt.xlim(GTSAgent.VIS_RANGE[0])
    # plt.ylim(GTSAgent.VIS_RANGE[1])
    plt.colorbar()
    logger.log_plot(fig, fig_name, step)
    plt.close(fig)
    