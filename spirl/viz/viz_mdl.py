import os
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from spirl.train import *



class MDLVisualizer(ModelTrainer):

    def __init__(self, args):
        self.args = args
        self.setup_device()

        # set up params
        self.conf = conf = self.get_config()

        self._hp = self._default_hparams()
        self._hp.overwrite(conf.general)  # override defaults with config file
        self._hp.exp_path = make_path(conf.exp_dir, args.path, args.prefix, args.new_dir)

        self._hp.batch_size = 2
        self.log_dir = log_dir = os.path.join(self._hp.exp_path, 'events')
        print('using log dir: ', log_dir)

        self.conf = self.postprocess_conf(conf)

        train_params = AttrDict(logger_class=self._hp.logger,
                                model_class=self._hp.model,
                                n_repeat=self._hp.epoch_cycles_train,
                                dataset_size=-1)
        self.model, self.loader = self.build_vizer(train_params, 'viz')
        self.optimizer = self.get_optimizer_class()(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self._hp.lr)
        
        print('conf.ckpt_path', conf.ckpt_path)
        if args.resume or conf.ckpt_path is not None:
            start_epoch = self.resume(args.resume, conf.ckpt_path)

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
        # sample_batched = self.loader.dataset[0]
        
        for batch_idx, sample_batched in enumerate(self.loader):
            inputs = AttrDict(map_dict(lambda x: x.to(self.device), sample_batched))
            output = self.model(inputs)
            output_prior = self.model(inputs, use_learned_prior=True)

            input_actions = to_numpy(inputs.actions)
            output_reconstruction = to_numpy(output.reconstruction)
            output_prior_recon = to_numpy(output_prior.reconstruction)

            input_actions = self.loader.dataset.action_scaler.inverse_transform(input_actions)
            output_reconstruction = self.loader.dataset.action_scaler.inverse_transform(output_reconstruction)
            output_prior_recon = self.loader.dataset.action_scaler.inverse_transform(output_prior_recon)

            print("=============== index", batch_idx)
            # print('input', inputs.actions[0])
            # print('output', output.reconstruction[0])
            # plots(to_numpy(inputs.actions[0]), to_numpy(output.reconstruction[0]))
            n = 0

            plots(input_actions[n], output_reconstruction[n], output_prior_recon[n])
            if batch_idx > 5:
                break
        print('finish')


def plots(input, output, out_prior):

    # print('input', input)
    # print('output', output)
    
    rad2deg = 180.0 / np.pi

    plt.figure(figsize=(10,4))
    titles = ['steering angle (deg)', 'pedal command']

    plt.subplot(1,2, 1)
    plt.plot(input[:,0] * rad2deg, 'b')
    plt.plot(output[:,0] *rad2deg, 'r')
    plt.plot(out_prior[:,0] *rad2deg, 'g')
    plt.title(titles[0])

    plt.subplot(1,2, 2)
    plt.plot(input[:,1], 'b', label='input action series')
    plt.plot(output[:,1], 'r', label='output reconstruction')
    plt.plot(out_prior[:,1], 'g', label='prior')
    plt.title(titles[1])

    plt.legend()

    plt.show()

def to_numpy(t):
    return t.cpu().detach().numpy()

if __name__ == '__main__':
    MDLVisualizer(args=get_args())
