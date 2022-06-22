from spirl.utils.general_utils import AttrDict

from spirl.components.data_loader import GlobalSplitVideoDataset, GTSDataset, UniformSeqDataset


from spirl.utils.gts_utils import state_dim

data_spec = AttrDict(
    # dataset_class=GTSDataset,
    dataset_class = UniformSeqDataset,
    n_actions=2,
    state_dim=state_dim,
    split=AttrDict(train=0.8, val=0.2, test=0.0),
    res=32,
    crop_rand_subseq=True,
    smooth_actions = True,
)
data_spec.max_seq_len = 300
