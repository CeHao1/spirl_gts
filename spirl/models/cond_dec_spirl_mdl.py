import torch
import torch.nn as nn

from spirl.utils.general_utils import batch_apply, ParamDict
from spirl.utils.pytorch_utils import get_constant_parameter, ResizeSpatial, RemoveSpatial
from spirl.models.skill_prior_mdl import SkillPriorMdl
from spirl.modules.subnetworks import Predictor, BaseProcessingLSTM, Encoder
from spirl.modules.variational_inference import MultivariateGaussian
from spirl.modules.variational_inference import MultivariateGaussian
from spirl.components.checkpointer import load_by_key, freeze_modules


class CDSPiRLMdl(SkillPriorMdl):
    """SPiRL model with conditional low-level skill decoder."""
    def build_network(self):
        assert not self._hp.use_convs  # currently only supports non-image inputs
        # assert self._hp.cond_decode    # need to decode based on state for closed-loop low-level
        self.q = self._build_inference_net()
        self.decoder = Predictor(self._hp,
                                 input_size=self.enc_size + self._hp.nz_vae,
                                 output_size=self._hp.action_dim,
                                 mid_size=self._hp.nz_mid_prior)
        self.p = self._build_prior_ensemble()
        self.log_sigma = get_constant_parameter(0., learnable=False)

    def decode(self, z, cond_inputs, steps, inputs=None):
        assert inputs is not None       # need additional state sequence input for full decode
        seq_enc = self._get_seq_enc(inputs)
        decode_inputs = torch.cat((seq_enc[:, :steps], z[:, None].repeat(1, steps, 1)), dim=-1)
        return batch_apply(decode_inputs, self.decoder)

    def _build_inference_net(self):
        # condition inference on states since decoder is conditioned on states too
        input_size = self._hp.action_dim
        return torch.nn.Sequential(
            BaseProcessingLSTM(self._hp, in_dim=input_size, out_dim=self._hp.nz_enc),
            torch.nn.Linear(self._hp.nz_enc, self._hp.nz_vae * 2)
        )

    def _run_inference(self, inputs):
        # run inference with state sequence conditioning
        # inf_input = torch.cat((inputs.actions, self._get_seq_enc(inputs)), dim=-1)
        inf_input = inputs.actions
        return MultivariateGaussian(self.q(inf_input)[:, -1])

    def _get_seq_enc(self, inputs):
        return inputs.states[:, :-1]

    def load_weights_and_freeze(self):
        """Optionally loads weights for components of the architecture + freezes these components."""
        if self._hp.embedding_checkpoint is not None:
            print("Loading pre-trained embedding from {}!".format(self._hp.embedding_checkpoint))
            self.load_state_dict(load_by_key(self._hp.embedding_checkpoint, 'decoder', self.state_dict(), self.device))
            self.load_state_dict(load_by_key(self._hp.embedding_checkpoint, 'q', self.state_dict(), self.device))
            freeze_modules([self.decoder, self.q])


    @property
    def enc_size(self):
        return self._hp.state_dim