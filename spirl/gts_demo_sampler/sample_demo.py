
import imp

from spirl.utils.general_utils import AttrDict, ParamDict
from spirl.train import set_seeds, make_path, datetime_str, save_config, get_exp_dir
from spirl.components.checkpointer import CheckpointHandler, save_cmd, save_git, get_config_path
from spirl.gts_demo_sampler.param import get_args

class SampleDemo:
    def __init__(self, args):
        self.args = args
        
        # set up params
        self.conf = self.get_config()
        self._hp = self._default_hparams()
        self._hp.overwrite(self.conf.general)

        self.init = self._hp.init(self.conf.init)
        self.start = self._hp.start(self.conf.start)
        self.done = self._hp.done(self.conf.done)
        self.sample = self._hp.sample(self.conf.sample)
        self.file = self._hp.file(self.conf.file)


    def _default_hparams(self):
        default_dict = ParamDict({
            'do_init': True,
            'start_num_epoch': 0,
            'num_epochs':   200,
        })
        return default_dict

    def sample_rollout(self):
        if self._hp.do_init:
            self.init.init_gts()
        for epoch_index in range(self._hp.start_num_epoch, self._hp.start_num_epoch + self._hp.num_epochs):
            print("index {}/{}".format(epoch_index, self._hp.num_epochs))
            self.sample_raw_data(epoch_index)
        
    def sample_raw_data(self, epoch_index):
        start_conditions = self.start.start_conditions
        done_function = self.done.done_function

        print('sample epoch {}'.format(epoch_index))
        raw_data = self.sample.sample_raw_data(start_conditions, done_function)
        file_name = 'raw_data_' + str(epoch_index)
        self.file.save_raw_data(raw_data, file_name)

    def convert_rollout(self):
        self.file.convert_to_rollout()

    def get_config(self):
        conf = AttrDict()

        # paths
        conf.exp_dir = get_exp_dir()
        conf.conf_path = get_config_path(self.args.path)

        print('loading from the config file {}'.format(conf.conf_path))
        conf_module = imp.load_source('conf', conf.conf_path)

        conf.general = conf_module.configuration
        conf.init = conf_module.init_config
        conf.start = conf_module.start_config
        conf.done = conf_module.done_config
        conf.sample = conf_module.sample_config
        conf.file = conf_module.file_config

        return conf


'''
for json args
"args":[
    "--path" , "spirl/configs/data_collect/gts/time_trial" ,
    "--prefix" , "demo_test"
]


'''

if __name__ == '__main__':
    sampler = SampleDemo(args=get_args())
    # sampler.sample_rollout()
    sampler.convert_rollout()

    # sampler.init.init_gts()
    # raw_data = sampler.sample_raw_data()
