from spirl.train import *

import os


class MDLVisualizer(ModelTrainer):

    def __init__(self, args):
        self.args = args
        self.setup_device()

        # set up params
        self.conf = conf = self.get_config()

        self._hp = self._default_hparams()
        self._hp.overwrite(conf.general)  # override defaults with config file
        self._hp.exp_path = make_path(conf.exp_dir, args.path, args.prefix, args.new_dir)
        self.log_dir = log_dir = os.path.join(self._hp.exp_path, 'events')
        print('using log dir: ', log_dir)

        self.conf = self.postprocess_conf(conf)

        train_params = AttrDict(logger_class=self._hp.logger,
                                model_class=self._hp.model,
                                n_repeat=self._hp.epoch_cycles_train,
                                dataset_size=-1)
        self.model, self.train_loader = self.build_vizer(train_params, 'viz')

        print('get model and data')
        self.test_once()

        

    def build_vizer(self, params, phase):

        model = params.model_class(self.conf.model)
        if torch.cuda.device_count() > 1:
            print("\nUsing {} GPUs!\n".format(torch.cuda.device_count()))
            model = DataParallelWrapper(model)
        model = model.to(self.device)
        model.device = self.device
        loader = self.get_dataset(self.args, model, self.conf.data, phase, params.n_repeat, params.dataset_size)
        return model, loader

    def test_once(self):
        for self.batch_idx, sample_batched in enumerate(self.train_loader):
            print(sample_batched)


if __name__ == '__main__':
    MDLVisualizer(args=get_args())
