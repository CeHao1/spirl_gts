import os
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from spirl.train import *

import seaborn as sns
from tqdm import tqdm
from spirl.utils.pytorch_utils import map2np

class MDLVisualizer(ModelTrainer):

    def __init__(self, args):
        self.args = args
        self.setup_device()

        # set up params
        self.conf = conf = self.get_config()

        self._hp = self._default_hparams()
        self._hp.overwrite(conf.general)  # override defaults with config file
        self._hp.exp_path = make_path(conf.exp_dir, args.path, args.prefix, args.new_dir)

        # self data dir
        self._hp.data_dir = '/home/msc/cehao/github_space/spirl_gts/save_rollout'
        self._hp.batch_size = 8
        
        self.log_dir = log_dir = os.path.join(self._hp.exp_path, 'events')
        print('using log dir: ', log_dir)

        self.conf = self.postprocess_conf(conf)

        train_params = AttrDict(logger_class=self._hp.logger,
                                model_class=self._hp.model,
                                n_repeat=self._hp.epoch_cycles_train,
                                dataset_size=-1)
        self.model, self.loader = self.build_vizer(train_params, 'train')
        self.optimizer = self.get_optimizer_class()(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self._hp.lr)
        
        print('conf.ckpt_path', conf.ckpt_path)
        if args.resume or conf.ckpt_path is not None:
            start_epoch = self.resume(args.resume, conf.ckpt_path)


        # self.model.switch_to_prior()
        print('get model and data')
        # self.show_value_distribution()
        self.show_one_value()
           

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
            # plots_distribution(*self.get_data(all_data=True))

    def show_value_distribution(self):
        # for idx in tqdm(range(10)):
        output = self.get_data(get_output=True)
        plot_z_mean_var(output)
        

    def get_data(self, all_data=False, get_output=False):
        # sample_batched = self.loader.dataset[0]
        
        for batch_idx, sample_batched in enumerate(self.loader):
            inputs = AttrDict(map_dict(lambda x: x.to(self.device), sample_batched))
 

            # direct data is z~encoder(), a~decoder(z)    
            output = self.model(inputs)

            print('!!inputs.states', inputs.states.shape)
            # output.RL_prior = [self.model.compute_learned_prior(s[0], first_only=True) for s in inputs.states[:,0]]
            output.RL_prior = self.model.compute_learned_prior( inputs.states[:,0, :], first_only=True)
            # output.RL_prior = self.model.compute_learned_prior(inputs.states[:,0])

            '''
            # use prior 
            self.model.switch_to_prior()
            output_prior = self.model(inputs, use_learned_prior=True)
            self.model.switch_to_inference()
            '''

            input_actions = to_numpy(inputs.actions)
            output_reconstruction = to_numpy(output.reconstruction)

            # output_prior_recon = to_numpy(output_prior.reconstruction)
            output_prior_recon = to_numpy(output.prior_reconstruction)

            break
        # print('finish')
        if get_output:
            return output
        
        if all_data:   
            def flat(x):
                return x.reshape(-1, x.shape[-1])                                                     
            return flat(input_actions), flat(output_reconstruction), flat(output_prior_recon)

        n = 0
        return input_actions[n], output_reconstruction[n], output_prior_recon[n]


def plot_z_mean_var(output):
    qs = (output.q)
    q_hats = (output.q_hat)
    # shape (batch, guassian)

    q_means = np.array(map2np([q.mean for q in qs]))
    q_vars = np.array(map2np([q.sigma for q in qs]))
    qhat_means = np.array(map2np([q.mean for q in q_hats]))
    qhat_vars = np.array(map2np([q.sigma for q in q_hats]))

    RL_qhat_means = np.array(map2np([q.mean for q in output.RL_prior]))
    RL_qhat_vars = np.array(map2np([q.sigma for q in output.RL_prior]))

    print('q means', q_means[0].shape)

    for idx in range(q_means[0].shape[0]):
        print('dim', idx)
        plt.figure(figsize=(14,4))
        plt.subplot(1,2,1)
        plt.plot(q_means[:, idx], 'b.', label='encoder')
        plt.plot(qhat_means[:, idx], 'r.', label='prior')
        # plt.plot(RL_qhat_means, 'r.')

        plt.grid()
        plt.title('latent variable distribution mean')
        plt.legend()

        plt.subplot(1,2,2)
        plt.plot(q_vars[:, idx], 'b.')
        plt.plot(qhat_vars[:, idx], 'r.')
        # plt.plot(RL_qhat_vars, 'r.')
        plt.grid()
        plt.title('latent variable distribution variance')

        plt.show()




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
    plt.plot(out_prior[:,0] *rad2deg, 'g')
    plt.title(titles[0], fontsize=15)
    # plt.ylim([-1.1, 1.1])

    plt.subplot(1,2, 2)
    plt.plot(input[:,1], 'b', label='input action series')
    plt.plot(output[:,1], 'r', label='output reconstruction')
    plt.plot(out_prior[:,1], 'g', label='prior')
    plt.title(titles[1], fontsize=15)
    # plt.ylim([-1.1, 1.1])

    plt.legend(fontsize=15)

    plt.show()

def plots_distribution(input, output, out_prior):


    range2deg = 1

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
