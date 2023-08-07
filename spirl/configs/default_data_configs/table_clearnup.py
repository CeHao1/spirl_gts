from spirl.utils.general_utils import AttrDict
from spirl.components.data_loader import GlobalSplitVideoDataset

data_spec = AttrDict(
    dataset_class=GlobalSplitVideoDataset,
    n_actions=4,
    state_dim=25,
    crop_rand_subseq=True,
)
data_spec.max_seq_len = 280