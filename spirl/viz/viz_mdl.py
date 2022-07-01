import os
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from spirl.train import *

import seaborn as sns
from tqdm import tqdm


class MDLVisualizer(ModelTrainer):

    def __init__(self, args):
        self.args = args
        self.setup_device()

        # set up params
        self.conf = conf = self.get_config()

        self._hp = self._default_hparams()
        self._hp.overwrite(conf.general)  # override defaults with config file
        self._hp.exp_path = make_path(conf.exp_dir, args.path, args.prefix, args.new_dir)

        self._hp.batch_size = 128
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


        # self.model.switch_to_prior()
        print('get model and data')
        self.show_one_value()
        # self.show_value_distribution()
        

    def build_vizer(self, params, phase):

        model = params.model_class(self.conf.model)
        if torch.cuda.device_count() > 1:
            print("\nUsing {} GPUs!\n".format(torch.cuda.device_count()))
            model = DataParallelWrapper(model)
        model = model.to(self.device)
        model.device = self.device
        loader = self.get_dataset(self.args, model, self.conf.data, phase, params.n_repeat, params.dataset_size)
        return model, loader


    def show_one_value(self):
        for idx in range(20):
            plots(*self.get_data())

    def show_value_distribution(self):
        inpt_mean = []
        oupt_mean = []
        prior_mean = []
        # for idx in tqdm(range(10)):
        inpt, oupt, prior = self.get_data(all_data=True)

        # inpt_mean.append(np.mean(inpt, axis=0))
        # oupt_mean.append(np.mean(oupt, axis=0))
        # prior_mean.append(np.mean(prior, axis=0))

        inpt_mean = np.mean(inpt, axis=1)
        oupt_mean = np.mean(oupt, axis=1)
        prior_mean = np.mean(prior, axis=1)
        plots_distribution(inpt_mean, oupt_mean, prior_mean)

    def get_data(self, all_data=False):
        # sample_batched = self.loader.dataset[0]
        
        for batch_idx, sample_batched in enumerate(self.loader):
            inputs = AttrDict(map_dict(lambda x: x.to(self.device), sample_batched))

            # self.model.switch_to_prior()            
            output = self.model(inputs)

            self.model.switch_to_prior()
            output_prior = self.model(inputs, use_learned_prior=True)
            self.model.switch_to_inference()

            input_actions = to_numpy(inputs.actions)
            output_reconstruction = to_numpy(output.reconstruction)
            output_prior_recon = to_numpy(output_prior.reconstruction)

            # input_actions = self.loader.dataset.action_scaler.inverse_transform(input_actions)
            # output_reconstruction = self.loader.dataset.action_scaler.inverse_transform(output_reconstruction)
            # output_prior_recon = self.loader.dataset.action_scaler.inverse_transform(output_prior_recon)

            break
        # print('finish')
        if all_data:
            return input_actions, output_reconstruction, output_prior_recon
        else:
            n = 0
            return input_actions[n], output_reconstruction[n], output_prior_recon[n]


def plots(input, output, out_prior):

    # print('input', input)
    # print('output', output)
    
    # rad2deg = 180.0 / np.pi
    rad2deg = 1

    plt.figure(figsize=(15,5))
    titles = ['steering angle', 'pedal command']

    plt.subplot(1,2, 1)
    plt.plot(input[:,0] * rad2deg, 'b')
    plt.plot(output[:,0] *rad2deg, 'r')
    # plt.plot(out_prior[:,0] *rad2deg, 'g')
    plt.title(titles[0])
    # plt.ylim([-1.1, 1.1])

    plt.subplot(1,2, 2)
    plt.plot(input[:,1], 'b', label='input action series')
    plt.plot(output[:,1], 'r', label='output reconstruction')
    # plt.plot(out_prior[:,1], 'g', label='prior')
    plt.title(titles[1])
    # plt.ylim([-1.1, 1.1])

    plt.legend()

    plt.show()

def plots_distribution(input, output, out_prior):


    range2deg = 30

    titles = ['steering angle (deg)', 'pedal command']


    # prior distribution
    # plt.figure(figsize=(15,5))

    # plt.subplot(1,2, 1)
    # sns.kdeplot(output[:,0] * range2deg, label='decoded steering')
    # sns.kdeplot(out_prior[:,0] * range2deg, label='prior')
    # plt.title('steering (deg)', fontsize=20)
    # plt.ylabel('density', fontsize=20)
    # plt.xlabel('steering angles degree', fontsize=20)
    # # plt.legend(fontsize=18)

    # plt.subplot(1,2, 2)
    # sns.kdeplot(output[:,1], label='decoded pedal')
    # sns.kdeplot(out_prior[:,1], label='prior')
    # plt.ylabel('density', fontsize=20)
    # plt.xlabel('pedal', fontsize=20)
    # plt.title('pedal', fontsize=20)

    # plt.legend(fontsize=18)
    # plt.show()



    # density
    plt.figure(figsize=(15,5))

    plt.subplot(1,2, 1)
    sns.kdeplot(input[:,0] * range2deg, label='input action series')
    sns.kdeplot(output[:,0] * range2deg, label='output reconstruction')
    sns.kdeplot(out_prior[:,0] * range2deg, label='prior')
    plt.title(titles[0], fontsize=20)
    plt.ylabel('density', fontsize=20)
    plt.xlabel('steering angles degree', fontsize=20)
    # plt.legend(fontsize=20)

    plt.subplot(1,2, 2)
    sns.kdeplot(input[:,1], label='input action series')
    sns.kdeplot(output[:,1], label='output reconstruction')
    sns.kdeplot(out_prior[:,1], label='prior')
    plt.ylabel('density', fontsize=20)
    plt.xlabel('pedal', fontsize=20)
    plt.title(titles[1], fontsize=20)

    plt.legend(fontsize=18)
    plt.show()

    # histogram
    plt.figure(figsize=(15,5))

    plt.subplot(1,2, 1)
    sns.histplot(input[:,0] * range2deg, color='b')
    sns.histplot(output[:,0] * range2deg, color='r')
    sns.histplot(out_prior[:,0] * range2deg,  color='g')
    plt.ylabel('counts', fontsize=20)
    plt.xlabel('steering angles degree', fontsize=20)
    plt.title(titles[0], fontsize=20)

    plt.subplot(1,2, 2)
    sns.histplot(input[:,1], label='input action series', color='b')
    sns.histplot(output[:,1], label='output reconstruction', color='r')
    sns.histplot(out_prior[:,1], label='prior', color='g')
    plt.ylabel('counts', fontsize=20)
    plt.xlabel('pedal', fontsize=20)
    plt.title(titles[1], fontsize=20)

    plt.legend(fontsize=20)
    plt.show()



def to_numpy(t):
    return t.cpu().detach().numpy()

if __name__ == '__main__':
    MDLVisualizer(args=get_args())
