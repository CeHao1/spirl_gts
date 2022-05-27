import os
import cv2
import h5py
import numpy as np
from spirl.utils.general_utils import AttrDict


class RolloutSaver(object):
    """Saves rollout episodes to a target directory."""
    def __init__(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.save_dir = save_dir
        self.counter = 0

    def save_rollout_to_file(self, episode):
        """Saves an episode to the next file index of the target folder."""
        # get save path
        save_path = os.path.join(self.save_dir, "rollout_{}.h5".format(self.counter))

        # save rollout to file
        f = h5py.File(save_path, "w")
        f.create_dataset("traj_per_file", data=1)

        # store trajectory info in traj0 group
        traj_data = f.create_group("traj0")
        traj_data.create_dataset("states", data=np.array(episode.observation))
        traj_data.create_dataset("images", data=np.array(episode.image, dtype=np.uint8))
        traj_data.create_dataset("actions", data=np.array(episode.action))
        traj_data.create_dataset("reward", data=np.array(episode.reward))
        traj_data.create_dataset("done", data=np.array(episode.done))

        terminals = np.array(episode.done)
        if np.sum(terminals) == 0:
            terminals[-1] = True

        # build pad-mask that indicates how long sequence is
        is_terminal_idxs = np.nonzero(terminals)[0]
        pad_mask = np.zeros((len(terminals),))
        pad_mask[:is_terminal_idxs[0]] = 1.
        traj_data.create_dataset("pad_mask", data=pad_mask)

        f.close()

        self.counter += 1

    def load_rollout_to_file(self, counter):
        # get save path
        save_path = os.path.join(self.save_dir, "rollout_{}.h5".format(counter))
        F = h5py.File(save_path, "r")
                
        data = AttrDict()
        key = "traj0"
        for name in F[key].keys():
                if name in ['states', 'actions', 'pad_mask', 'reward', 'done']:
                    data[name] = F[key + '/' + name][()].astype(np.float32)

        return data


    def _resize_video(self, images, dim=64):
        """Resize a video in numpy array form to target dimension."""
        ret = np.zeros((images.shape[0], dim, dim, 3))

        for i in range(images.shape[0]):
            ret[i] = cv2.resize(images[i], dsize=(dim, dim),
                                interpolation=cv2.INTER_CUBIC)

        return ret.astype(np.uint8)

    def reset(self):
        """Resets counter."""
        self.counter = 0
