import glob
import imp
import os
import random
import h5py
import numpy as np
import copy
import torch.utils.data as data
import itertools

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from spirl.utils.math_utils import positive2unit
import matplotlib.pyplot as plt

from spirl.utils.general_utils import AttrDict, ParamDict, map_dict, maybe_retrieve, shuffle_with_seed
from spirl.utils.pytorch_utils import RepeatedDataLoader
from spirl.utils.video_utils import resize_video
from spirl.utils.math_utils import smooth

# ================================= original video loader ======================
class Dataset(data.Dataset):

    def __init__(self, data_dir, data_conf, phase, shuffle=True, dataset_size=-1):
        self.phase = phase
        self.data_dir = data_dir
        self.spec = data_conf.dataset_spec
        self.dataset_size = dataset_size
        self.device = data_conf.device

        print('loading files from', self.data_dir)
        self.filenames = self._get_filenames()
        self.filenames = self._filter_filenames(self.filenames)
        self.samples_per_file = self._get_samples_per_file(self.filenames[0])

        self.shuffle = shuffle and phase == 'train'
        self.n_worker = 8 if shuffle else 1  # was 4 before

    def get_data_loader(self, batch_size, n_repeat):
        print('len {} dataset {}'.format(self.phase, len(self)))

        assert self.device in ['cuda', 'cpu']  # Otherwise the logic below is wrong
        return RepeatedDataLoader(self, batch_size=batch_size, shuffle=self.shuffle, num_workers=self.n_worker,
                                  drop_last=True, n_repeat=n_repeat, pin_memory=self.device == 'cuda',
                                  worker_init_fn=lambda x: np.random.seed(np.random.randint(65536) + x))

    def __getitem__(self, index):
        """Load a single sequence from disk according to index."""
        raise NotImplementedError("Needs to be implemented in sub-class!")

    def _get_samples_per_file(self, path):
        """Returns number of data samples per data file."""
        raise NotImplementedError("Needs to be implemented in sub-class!")

    def _get_filenames(self):
        """Loads filenames from self.data_dir, expects subfolders train/val/test, each with hdf5 files"""
        filenames = sorted(glob.glob(os.path.join(self.data_dir, self.phase + '/*.h5')))
        if not filenames:
            raise RuntimeError('No filenames found in {}'.format(self.data_dir))
        filenames = shuffle_with_seed(filenames)
        return filenames

    def _filter_filenames(self, filenames):
        """Optionally filters filenames / limits to max number of filenames etc."""
        if "n_seqs" in self.spec:
            # limit the max number of sequences in dataset
            if self.phase == "train" and len(filenames) < self.spec.n_seqs:
                raise ValueError("Not enough seqs in dataset!")
            filenames = filenames[:self.spec.n_seqs]

        if "seq_repeat" in self.spec:
            # repeat sequences in dataset
            repeat = max(self.spec.seq_repeat, self.dataset_size / len(filenames))
            filenames *= int(repeat)
            filenames = shuffle_with_seed(filenames)

        return filenames

    def __len__(self):
        if self.dataset_size != -1:
            return self.dataset_size
        return len(self.filenames) * self.samples_per_file


class GlobalSplitDataset(Dataset):
    """Splits in train/val/test using global percentages."""
    def _get_filenames(self):
        filenames = self._load_h5_files(self.data_dir)

        if not filenames:
            raise RuntimeError('No filenames found in {}'.format(self.data_dir))
        filenames = shuffle_with_seed(filenames)
        if (self.phase != 'viz'):
            filenames = self._split_with_percentage(self.spec.split, filenames)
        return filenames

    def _load_h5_files(self, dir):
        filenames = []
        for root, dirs, files in os.walk(dir):
            for file in files:
                if file.endswith(".h5"): filenames.append(os.path.join(root, file))
        return filenames

    def _split_with_percentage(self, frac, filenames):
        assert sum(frac.values()) <= 1.0  # fractions cannot sum up to more than 1
        assert self.phase in frac
        if self.phase == 'train':
            start, end = 0, frac['train']
        elif self.phase == 'val':
            start, end = frac['train'], frac['train'] + frac['val']
        else:
            start, end = frac['train'] + frac['val'], frac['train'] + frac['val'] + frac['test']
        start, end = int(len(filenames) * start), int(len(filenames) * end)
        return filenames[start:end]


class VideoDataset(Dataset):
    """Generic video dataset. Assumes that HDF5 file has images/states/actions/pad_mask."""
    def __init__(self, *args, resolution, **kwargs):
        super().__init__(*args, **kwargs)
        self.randomize_length = self.spec.randomize_length if 'randomize_length' in self.spec else False
        self.crop_subseq = 'crop_rand_subseq' in self.spec and self.spec.crop_rand_subseq
        self.img_sz = resolution
        self.subsampler = self._get_subsampler()

    def __getitem__(self, index):
        data = self._get_raw_data(index)

        # maybe subsample seqs
        if self.subsampler is not None:
            data = self._subsample_data(data)

        # sample random subsequence of fixed length
        if self.crop_subseq:
            end_ind = np.argmax(data.pad_mask * np.arange(data.pad_mask.shape[0], dtype=np.float32), 0)
            data = self._crop_rand_subseq(data, end_ind, length=self.spec.subseq_len)


        # Make length consistent
        start_ind = 0
        end_ind = np.argmax(data.pad_mask * np.arange(data.pad_mask.shape[0], dtype=np.float32), 0) \
            if self.randomize_length or self.crop_subseq else self.spec.max_seq_len - 1
        end_ind, data = self._sample_max_len_video(data, end_ind, target_len=self.spec.subseq_len if self.crop_subseq
                                                                                  else self.spec.max_seq_len)

        if self.randomize_length:
            end_ind = self._randomize_length(start_ind, end_ind, data)
            data.start_ind, data.end_ind = start_ind, end_ind

        # perform final processing on data
        data.images = self._preprocess_images(data.images)

        return data

    def _get_raw_data(self, index):
        data = AttrDict()
        file_index = index // self.samples_per_file
        path = self.filenames[file_index]

        try:
            with h5py.File(path, 'r') as F:
                ex_index = index % self.samples_per_file  # get the index
                key = 'traj{}'.format(ex_index)

                # Fetch data into a dict
                for name in F[key].keys():
                    if name in ['states', 'actions', 'pad_mask']:
                        data[name] = F[key + '/' + name][()].astype(np.float32)

                if key + '/images' in F:
                    data.images = F[key + '/images'][()]
                else:
                    data.images = np.zeros((data.states.shape[0], 2, 2, 3), dtype=np.uint8)

            # print('data action', data.actions.shape, 'states', data.states.shape)
        except:
            raise ValueError("Could not load from file {}".format(path))
        return data

    def _get_samples_per_file(self, path):
        with h5py.File(path, 'r') as F:
            return F['traj_per_file'][()]

    def _get_subsampler(self):
        subsampler_class = maybe_retrieve(self.spec, 'subsampler')
        if subsampler_class is not None:
            subsample_args = maybe_retrieve(self.spec, 'subsample_args')
            assert subsample_args is not None  # need to specify subsampler args dict
            subsampler = subsampler_class(**subsample_args)
        else:
            subsampler = None
        return subsampler

    def _subsample_data(self, data_dict):
        idxs = None
        for key in data_dict:
            data_dict[key], idxs = self.subsampler(data_dict[key], idxs=idxs)
        return data_dict

    def _crop_rand_subseq(self, data, end_ind, length):
        """Crops a random subseq of specified length from the full sequence."""
        assert length <= end_ind + 1     # sequence needs to be longer than desired subsequence length
        start = np.random.randint(0, end_ind - length + 2)
        for key in data:
            data[key] = data[key][start : int(start+length)]
        return data

    def _sample_max_len_video(self, data_dict, end_ind, target_len):
        """ This function processes data tensors so as to have length equal to target_len
        by sampling / padding if necessary """
        extra_length = (end_ind + 1) - target_len
        if self.phase == 'train':
            offset = max(0, int(np.random.rand() * (extra_length + 1)))
        else:
            offset = 0

        data_dict = map_dict(lambda tensor: self._maybe_pad(tensor, offset, target_len), data_dict)
        if 'actions' in data_dict:
            data_dict.actions = data_dict.actions[:-1]
        end_ind = min(end_ind - offset, target_len - 1)

        return end_ind, data_dict

    @staticmethod
    def _maybe_pad(val, offset, target_length):
        """Pads / crops sequence to desired length."""
        val = val[offset:]
        len = val.shape[0]
        if len > target_length:
            return val[:target_length]
        elif len < target_length:
            return np.concatenate((val, np.zeros([int(target_length - len)] + list(val.shape[1:]), dtype=val.dtype)))
        else:
            return val

    def _randomize_length(self, start_ind, end_ind, data_dict):
        """ This function samples part of the input tensors so that the length of the result
        is uniform between 1 and max """

        length = 3 + int(np.random.rand() * (end_ind - 2))  # The length of the seq is from 2 to total length
        chop_length = int(np.random.rand() * (end_ind + 1 - length))  # from 0 to the reminder
        end_ind = length - 1
        pad_mask = np.logical_and((np.arange(self.spec['max_seq_len']) <= end_ind),
                                  (np.arange(self.spec['max_seq_len']) >= start_ind)).astype(np.float32)

        # Chop off the beginning of the arrays
        def pad(array):
            array = np.concatenate([array[chop_length:], np.repeat(array[-1:], chop_length, 0)], 0)
            array[end_ind + 1:] = 0
            return array

        for key in filter(lambda key: key != 'pad_mask', data_dict):
            data_dict[key] = pad(data_dict[key])
        data_dict.pad_mask = pad_mask

        return end_ind

    def _preprocess_images(self, images):
        assert images.dtype == np.uint8, 'image need to be uint8!'
        images = resize_video(images, (self.img_sz, self.img_sz))
        images = np.transpose(images, [0, 3, 1, 2])  # convert to channel-first
        images = images.astype(np.float32) / 255 * 2 - 1
        assert images.dtype == np.float32, 'image need to be float32!'
        return images


class PreloadVideoDataset(VideoDataset):
    """Loads all sequences into memory for accelerated training (only possible for small datasets)."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._data = self._load_data()

    def _load_data(self):
        """Load all sequences into memory."""
        print("Preloading all sequences from {}".format(self.data_dir))
        return [super(PreloadVideoDataset, self)._get_raw_data(i) for i in range(len(self.filenames))]

    def _get_raw_data(self, index):
        return self._data[index]


class GlobalSplitVideoDataset(VideoDataset, GlobalSplitDataset):
    pass

class PreloadGlobalSplitVideoDataset(PreloadVideoDataset, GlobalSplitDataset):
    pass


class SmoothDataset(GlobalSplitVideoDataset):
    def smooth_actions(self, data, length):
        # odd length
        length = length//2 * 2 + 1
        for idx in range(data.actions.shape[1]):
            # clip to [-1,1]
            data.actions[:, idx] = np.clip(data.actions[:, idx], -1.0, 1.0)
            data.actions[:, idx] = smooth(data.actions[:, idx], length)
        return data

    def _get_raw_data(self, index):
        data = super()._get_raw_data(index)
        data = self.smooth_actions(data, self.spec.subseq_len)
        return data

# ===============================================
class GTSDataset(GlobalSplitVideoDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.smooth = 'smooth_actions' in self.spec and self.spec.smooth_actions

        # file_path = os.path.join(os.environ["EXP_DIR"], "skill_prior_learning/gts/standard_table")
        
        # import pickle
        # if not os.path.exists(file_path):
        #     standard_table = self.standardlize()
        #     f = open(file_path, "wb")
        #     pickle.dump(standard_table, f)
        #     f.close()
        #     print('save standard_table')
        # else:
        #     f = open(file_path, "rb")
        #     standard_table = pickle.load(f)
        #     f.close()
        #     print('load standard_table')
        # self.state_scaler = standard_table['state']
        # self.action_scaler = standard_table['action']

        # print the standard table 
        # print('===== action scaler =====')
        # print(self.action_scaler.mean_, self.action_scaler.scale_)
        # test_actions = [[-1.0, -1.0],[1.0, 1.0]]
        # print('converted action range', self.action_scaler.inverse_transform(test_actions))

    def getitem_(self, index):
        data = self._get_raw_data(index)

        # smooth the action 
        if self.smooth:
            data = self.smooth_actions(data, self.spec.subseq_len)

        # try to modify actions
        # data = self.modify_actions(data)

        # maybe subsample seqs
        if self.subsampler is not None:
            data = self._subsample_data(data)

        # sample random subsequence of fixed length
        if self.crop_subseq:
            end_ind = np.argmax(data.pad_mask * np.arange(data.pad_mask.shape[0], dtype=np.float32), 0)
            data = self._crop_rand_subseq(data, end_ind, length=self.spec.subseq_len)


        # Make length consistent
        start_ind = 0
        end_ind = np.argmax(data.pad_mask * np.arange(data.pad_mask.shape[0], dtype=np.float32), 0) \
            if self.randomize_length or self.crop_subseq else self.spec.max_seq_len - 1
        end_ind, data = self._sample_max_len_video(data, end_ind, target_len=self.spec.subseq_len if self.crop_subseq
                                                                                  else self.spec.max_seq_len)

        if self.randomize_length:
            end_ind = self._randomize_length(start_ind, end_ind, data)
            data.start_ind, data.end_ind = start_ind, end_ind

        # print('action shape', data.actions.shape)
        # print('states shape', data.states.shape)

        return data

    def modify_actions(self, data):
        actions = data.actions

        # change the bias
        a0 = np.mean(actions, axis=0)

        a0_change_range = 5.0 
        # rand_seed = np.random.rand(*a0.shape) * 2 -1.0
        rand_seed = np.random.rand(*a0.shape)
        a0_change_rate = rand_seed * a0_change_range

        a0_new = a0 * a0_change_rate

        # change the manitude
        action_origin = actions - actions[0]

        r = np.clip(np.abs(a0_change_rate), 1.0, None) * np.sign(a0_change_rate)
        change_range = np.random.rand(*a0.shape) * r

        # final results
        actions_new = a0_new + action_origin * change_range
        actions_new = np.float32(actions_new)
        data.actions = np.clip(actions_new, -1, 1)

        return data

    def smooth_actions(self, data, length):
        # odd length
        length = length//2 * 2 + 1
        for idx in range(data.actions.shape[1]):
            # clip to [-1,1]
            data.actions[:, idx] = np.clip(data.actions[:, idx], -1.0, 1.0)
            data.actions[:, idx] = smooth(data.actions[:, idx], length)
        return data

    def standardlize(self):
        from tqdm import tqdm
        file_number = len(self)
        iterate_times = 1
        sampler_number = int(file_number)
        # sampler_number = 2

        data0 = super().__getitem__(0)
        # data_state_list = data0.states
        # data_action_list = data0.actions

        data_state_list = []
        data_action_list = []


        for sample_roll in range(iterate_times):
            print('sample roll is {}, total roll {}'.format(sample_roll, iterate_times))
            for i in tqdm(range(sampler_number)):
                
                data = self._get_raw_data(i) # use full length data
                data = self.smooth_actions(data, self.spec.subseq_len)

                # print('data0', data0.actions.shape, data0.states.shape, type(data0.actions))
                # print('data', data.actions.shape, data.states.shape, type(data.actions))

                data_state_list.append(data.states)
                data_action_list.append(data.actions)

                # data_state_list = np.concatenate((data_state_list, data.states), axis=0)
                # data_action_list = np.concatenate((data_action_list, data.actions), axis=0)

        data_state_list = np.concatenate(data_state_list, axis=0)
        data_action_list = np.concatenate(data_action_list, axis=0)

        # data_state_list = np.array(data_state_list)
        # data_action_list = np.array(data_action_list)
        state_shapes = data_state_list.shape
        action_shapes = data_action_list.shape

        # print('state_shapes', state_shapes, 'action_shapes', action_shapes)

        # data_state_list = data_state_list.reshape(state_shapes[0] * state_shapes[1], state_shapes[2])
        # data_action_list = data_action_list.reshape(action_shapes[0] * action_shapes[1], action_shapes[2])

        # convert steer to [0] by dividing pi/6
        # data_action_list[:,0] /= np.pi / 6

        state_scaler = StandardScaler()
        state_scaler.fit(data_state_list)

        # action_scaler = MinMaxScaler(feature_range=(-1, 1))
        # action_scaler.min_ = [0.0, 0.0]
        # action_scaler.scale_ = [1/3, 1.0]

        action_scaler = StandardScaler()
        # action_scaler.fit(data_action_list)

        action_scaler.mean_ = [0.0, 0.0]
        # action_scaler.scale_ = 1.0
        # action_scaler.scale_ = [0.5, 1.0]
        action_scaler.scale_ = [1.0, 1.0]

        standard_table = {
            'state' : state_scaler,
            'action': action_scaler
        }

        print("============= strandard =================")
        print('states:')
        print(state_scaler.mean_, state_scaler.scale_)

        print('actions:')
        # print(action_scaler.min_, action_scaler.scale_)
        print(action_scaler.mean_, action_scaler.scale_)

        print('action limits, min: {}, max: {}'.format(np.min(data_action_list, axis=0), np.max(data_action_list, axis=0)))

        return standard_table
        
    def __getitem__(self, item):
        data = self.getitem_(item)
        data.states = self.state_scaler.transform(data.states)
        data.actions = self.action_scaler.transform(data.actions)
        return data

class GlobalSplitStateSequenceDataset(GlobalSplitVideoDataset):
    """Outputs observation in data dict, not images."""
    def __getitem__(self, item):
        data = super().__getitem__(item)
        data.observations = data.pop('states')
        return data


class GlobalSplitActionSequenceDataset(GlobalSplitVideoDataset):
    """Outputs observation in data dict, not images."""
    def __getitem__(self, item):
        data = super().__getitem__(item)
        data.observations = data.pop('actions')
        return data


class MixedVideoDataset(GlobalSplitVideoDataset):
    """Loads filenames from multiple directories and merges them with percentage."""
    def _load_h5_files(self, unused_dir):
        assert 'data_dirs' in self.spec and 'percentages' in self.spec
        assert np.sum(self.spec.percentages) == 1
        files = [super(MixedVideoDataset, self)._load_h5_files(dir) for dir in self.spec.data_dirs]
        files = [shuffle_with_seed(f) for f in files]
        total_size = min([1 / p * len(f) for p, f in zip(self.spec.percentages, files)])
        filenames = list(itertools.chain.from_iterable(
            [f[:int(total_size*p)] for p, f in zip(self.spec.percentages, files)]))
        return filenames


class GeneratedVideoDataset(VideoDataset):
    def __len__(self):
        if self.dataset_size != -1:
            return self.dataset_size

        if self.phase == 'train':
            return 10000
        else:
            return 200

    def _get_filenames(self):
        return [None]

    def _get_samples_per_file(self, path):
        pass

    def get_sample(self):
        raise NotImplementedError("Needs to be implemented by child class.")

    def __getitem__(self, index):
        if not self.shuffle:
            # Set seed such that validation is always deterministic
            np.random.seed(index)
        data = self.get_sample()
        return data

    @staticmethod
    def visualize(*args, **kwargs):
        pass


class RandomVideoDataset(GeneratedVideoDataset):
    def get_sample(self):
        data_dict = AttrDict()
        data_dict.images = np.random.rand(self.spec['max_seq_len'], 3, self.img_sz, self.img_sz).astype(np.float32)
        data_dict.states = np.random.rand(self.spec['max_seq_len'], self.spec['state_dim']).astype(np.float32)
        data_dict.actions = np.random.rand(self.spec['max_seq_len'] - 1, self.spec['n_actions']).astype(np.float32)
        return data_dict


class CustomizedSeqDataset(GlobalSplitVideoDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.spec = args[1].dataset_spec
        self.raw_state_length = self.spec.subseq_len
        self.raw_action_length = self.spec.subseq_len - 1
        self._hp = self._get_hp()
        self._hp.overwrite(self.spec)

    def _get_hp(self):
        hp = ParamDict(
            num_of_slop_change = 1,
        )
        return hp

    def __getitem__(self, index):
        data = AttrDict()
        data.states = self._generate_empty_states()
        data.actions = self._generate_actions()
        return data

    def _generate_empty_states(self):
        states = np.zeros((self.raw_state_length, self.spec.state_dim))
        return np.float32(states)

    def _generate_actions(self):
        actions = []
        for idx in range(self.spec.n_actions):
            actions.append(self._generate_action_sequence())

        actions = np.array(actions)
        return np.float32(actions.T)

    def _generate_action_sequence(self):
        action = np.zeros(self.raw_action_length - 1)
        return action

    def _clip_to_one(self, v):
        return np.clip(v, -1.0, 1.0)

    def __len__(self):
        return int(1e5)

class UniformSeqDataset(CustomizedSeqDataset):
    def _generate_action_sequence(self):
        mean_value = positive2unit(np.random.rand())
        raw_seq_value = np.random.randn(self.raw_action_length) / 5.0
        seq_value = self._generate_cumulated_seq(mean_value, raw_seq_value)
        action = seq_value
        return action

    def _generate_cumulated_seq(self, mean_value, raw_seq):
        slope = 1 if np.random.rand() > 0.5 else -1
        pos_of_slop_change = np.random.choice(np.arange(self.raw_action_length), self._hp.num_of_slop_change)

        seq = [mean_value]
        for idx in range(self.raw_action_length):
            if idx in pos_of_slop_change:
                slope *= -1
            new_v = seq[-1] + np.abs(raw_seq[idx]) * slope
            seq.append( self._clip_to_one(new_v) )

        seq = np.array(seq[1:])
        return seq
        

if __name__ == "__main__":
    # from spirl.configs.default_data_configs.gts import data_spec
    from spirl.configs.skill_prior_learning.gts.hierarchical.conf import data_config
    data_dir = 'None'
    dataset = UniformSeqDataset(data_dir, data_config)
    d1 = dataset[1]

    
    # print(d1.states.shape, d1.actions.shape)
    actions = d1.actions
    states = d1.states

    print('action shape: {}, state shape: {}'.format(actions.shape, states.shape))

    plt.figure(figsize=(15, 6))
    plt.subplot(121)
    plt.plot(actions[:,0])

    plt.subplot(122)
    plt.plot(actions[:,1])
    plt.show()
    