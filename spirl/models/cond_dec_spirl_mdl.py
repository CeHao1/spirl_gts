import torch
import torch.nn as nn

from spirl.utils.general_utils import batch_apply, ParamDict
from spirl.utils.pytorch_utils import get_constant_parameter, make_one_hot
from spirl.models.skill_prior_mdl import SkillPriorMdl, ImageSkillPriorMdl
from spirl.modules.subnetworks import Predictor, BaseProcessingLSTM, Encoder
from spirl.modules.variational_inference import MultivariateGaussian
from spirl.components.checkpointer import load_by_key, freeze_modules


class CDSPiRLMdl(SkillPriorMdl):
    """SPiRL model with closed-loop low-level skill decoder."""
    def build_network(self):
        assert not self._hp.use_convs  # currently only supports non-image inputs
        assert self._hp.cond_decode    # need to decode based on state for closed-loop low-level
        self.q = self._build_inference_net()
        self.decoder = Predictor(self._hp,
                                 input_size=self.enc_size + self._hp.nz_vae,
                                 output_size= self.action_size * 2,
                                 mid_size=self._hp.nz_mid_prior)
        self.p = self._build_prior_ensemble()
        self._log_sigma = get_constant_parameter(0., learnable=False)

    def decode(self, z, cond_inputs, steps, inputs=None):
        # the decode only use for training, so here we use deterministic 
        
        assert inputs is not None       # need additional state sequence input for full decode
        seq_enc = self._get_seq_enc(inputs)
        decode_inputs = torch.cat((seq_enc[:, :steps], z[:, None].repeat(1, steps, 1)), dim=-1)

        output = batch_apply(decode_inputs, self.decoder)
        output = output[..., :self.action_size]
        return output

    def _build_inference_net(self):
        # condition inference on states since decoder is conditioned on states too
        # input_size = self._hp.action_dim + self.prior_input_size
        input_size  = self._hp.action_dim 
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

    def enc_obs(self, obs):
        """Optionally encode observation for decoder."""
        return obs

    def load_weights_and_freeze(self):
        """Optionally loads weights for components of the architecture + freezes these components."""
        if self._hp.embedding_checkpoint is not None:
            print("Loading pre-trained embedding from {}!".format(self._hp.embedding_checkpoint))
            self.load_state_dict(load_by_key(self._hp.embedding_checkpoint, 'decoder', self.state_dict(), self.device))
            self.load_state_dict(load_by_key(self._hp.embedding_checkpoint, 'q', self.state_dict(), self.device))
            freeze_modules([self.decoder, self.q])
        else:
            super().load_weights_and_freeze()

    @property
    def enc_size(self):
        return self._hp.state_dim

    @property
    def action_size(self):
        return self._hp.action_dim


class TimeIndexCDSPiRLMDL(CDSPiRLMdl):
    # decoder input (s, z, idx)

    def build_network(self):
        assert not self._hp.use_convs  # currently only supports non-image inputs
        assert self._hp.cond_decode    # need to decode based on state for closed-loop low-level
        self.q = self._build_inference_net()
        self.decoder = Predictor(self._hp,
                                 input_size=self.enc_size + self._hp.nz_vae + self._hp.n_rollout_steps, 
                                 output_size= self.action_size * 2,
                                 mid_size=self._hp.nz_mid_prior)
        self.p = self._build_prior_ensemble()

    def decode(self, z, cond_inputs, steps, inputs=None):
        # the decode only use for training, so here we use deterministic 
        
        assert inputs is not None       # need additional state sequence input for full decode
        seq_enc = self._get_seq_enc(inputs)

        idx = torch.tensor(torch.arange(steps), device=self.device)
        one_hot = make_one_hot(idx, steps).repeat(seq_enc.shape[0], 1, 1)
        decode_inputs = torch.cat((seq_enc[:, :steps], z[:, None].repeat(1, steps, 1), one_hot), dim=-1)

        # print('='*20)
        # print('seq_enc', seq_enc.shape, 'z', z.shape)
        # print('seq_enc[:, :steps]', seq_enc[:, :steps].shape)
        # print('z repeat', z[:, None].repeat(1, steps, 1).shape)
        # print('decode_inputs',  decode_inputs.shape)

        # # print('idx', idx.shape)
        # print('one hot', one_hot.shape)

        output = batch_apply(decode_inputs, self.decoder)
        output = output[..., :self.action_size]
        return output
