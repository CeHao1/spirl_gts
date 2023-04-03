
from spirl.rl.agents.ac_agent import SACAgent
from spirl.rl.agents.prior_sac_agent import ActionPriorSACAgent
from spirl.rl.agents.skill_critic.hl_inherit_agent  import HLInheritAgent

from spirl.rl.envs.gts_corner2.gts_corner2_single import GTSEnv_Corner2_Single

from spirl.utils.general_utils import AttrDict
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class GTSAgent:

    VIS_RANGE = GTSEnv_Corner2_Single.VIS_RANGE
    START_POS = GTSEnv_Corner2_Single.START_POS
    TARGET_POS = GTSEnv_Corner2_Single.TARGET_POS

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
        key_list = ['pos[0]', 'pos[2]', 'vx']
        info_batch = AttrDict(
            observation = np.array([[info_t[0]['state'][key] for key in key_list] for info_t in info])
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
        self.info_replay_buffer.append(self._select_info(experience_batch['info']))
        super().add_experience(experience_batch)

    def _vis_hl_q(self, logger, step):
        """Visualizes high-level Q function."""
        self._vis_q(logger, step, plot_type='gts')

class GTSHLInerientAgent(HLInheritAgent, GTSAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.info_replay_buffer = self._hp.replay(self._hp.replay_params)

    def visualize(self, logger, rollout_storage, step):
        GTSAgent.visualize(self, logger, rollout_storage, step)
        self._vis_hl_q(logger, step)

    def add_experience(self, experience_batch):
        self.info_replay_buffer.append(self._select_info(experience_batch['info']))
        super().add_experience(experience_batch)

    def _vis_hl_q(self, logger, step):
        """Visualizes high-level Q function."""
        self._vis_q(logger, step, plot_type='gts')



def plot_gts_traj(states, logger, step, size):

    # plot replay with velocity
    fig = plt.figure(figsize=(8,8))
    plt.scatter(states[:, 0], states[:, 1], c=states[:, 2], cmap='Reds', s=1)
    # plt.plot(GTSAgent.START_POS[0], GTSAgent.START_POS[1], 'go')
    # plt.plot(GTSAgent.TARGET_POS[0], GTSAgent.TARGET_POS[1], 'ro')

    plt.axis("equal")
    plt.title('replay, step ' + str(step) + ' size ' + str(size))
    # plt.xlim(GTSAgent.VIS_RANGE[0])
    # plt.ylim(GTSAgent.VIS_RANGE[1])
    logger.log_plot(fig, "velocity_vis", step)
    plt.close(fig)


    # plot density
    # dist = states[:, :2] - GTSAgent.START_POS  # TODO dist from start line 
    # dist_to_start = np.sqrt(dist[:,0]**2 + dist[:,1]**2)
    # show_index = np.where(dist_to_start > 1.0)[0]

    show_index = np.arange(states.shape[0])

    fig = plt.figure(figsize=(10,8))
    sns.histplot(x=states[show_index, 0], y=states[show_index, 1], cmap='Blues', cbar=True,
                 bins=50, pthresh=0.01)
    plt.plot(GTSAgent.START_POS[0], GTSAgent.START_POS[1], 'go')
    plt.plot(GTSAgent.TARGET_POS[0], GTSAgent.TARGET_POS[1], 'ro')
    plt.axis("equal")
    plt.title('density, step ' + str(step) + ' size ' + str(size))
    plt.xlim(GTSAgent.VIS_RANGE[0])
    plt.ylim(GTSAgent.VIS_RANGE[1])
    logger.log_plot(fig, "density_vis", step)
    plt.close(fig)


def plot_gts_value(q, states, logger, step, size, fig_name='vis'):
    fig = plt.figure(figsize=(10,8))
    plt.scatter(states[:, 0], states[:, 1], s=5, c=q, cmap='Oranges')
    plt.plot(GTSAgent.START_POS[0], GTSAgent.START_POS[1], 'go')
    plt.plot(GTSAgent.TARGET_POS[0], GTSAgent.TARGET_POS[1], 'ro')
    plt.axis("equal")
    plt.title(fig_name + ' step ' + str(step) + ' size ' + str(size))
    plt.xlim(GTSAgent.VIS_RANGE[0])
    plt.ylim(GTSAgent.VIS_RANGE[1])
    plt.colorbar()
    logger.log_plot(fig, fig_name, step)
    plt.close(fig)
    