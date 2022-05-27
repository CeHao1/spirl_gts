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

        # have a look at the replay buffer

        