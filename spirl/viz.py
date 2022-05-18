import os
from spirl.rl.train import RLTrainer
from spirl.rl.components.params import get_args
from spirl.train import  make_path

class Visualizer(RLTrainer):

    def __init__(self, args):
        self.args = args
        self.setup_device()
        self.conf = self.get_config()

        self._hp = self._default_hparams()
        self._hp.overwrite(self.conf.general)  # override defaults with config file
        self._hp.exp_path = make_path(self.conf.exp_dir, args.path, args.prefix, args.new_dir)
        self.log_dir = log_dir = os.path.join(self._hp.exp_path, 'log')
        print('using log dir: ', log_dir)

        self.agent = self._hp.agent(self.conf.agent)
        self.agent.to(self.device)

        print('set up the viz')

if __name__ == '__main__':
    Visualizer(args=get_args())

