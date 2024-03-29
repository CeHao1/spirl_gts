import torch
from torch.nn import BCEWithLogitsLoss

from spirl.utils.general_utils import AttrDict, get_dim_inds
from spirl.modules.variational_inference import Gaussian

from spirl.utils.pytorch_utils import map2np, map2torch
import numpy as np

class Loss():
    def __init__(self, weight=1.0, breakdown=None):
        """
        
        :param weight: the balance term on the loss
        :param breakdown: if specified, a breakdown of the loss by this dimension will be recorded
        """
        self.weight = weight
        self.breakdown = breakdown
    
    def __call__(self, *args, weights=1, reduction='mean', store_raw=False, separate_dim=False, **kwargs):
        """

        :param estimates:
        :param targets:
        :return:
        """
        # weights = self.weight
        error = self.compute(*args, **kwargs) 
        if weights is not 1:
            weights = np.array(weights)
            assert error.shape[-1] == weights.shape[-1]
            for idx in range(error.shape[-1]):
                error[:,:,idx] *= weights[idx]

        if reduction != 'mean':
            raise NotImplementedError

        loss = AttrDict(value=error.mean(), weight=self.weight)

        if self.breakdown is not None:
            reduce_dim = get_dim_inds(error)[:self.breakdown] + get_dim_inds(error)[self.breakdown+1:]
            loss.breakdown = error.detach().mean(reduce_dim) if reduce_dim else error.detach()

        if store_raw:
            loss.error_mat = error.detach()

        if separate_dim:
            error_separate = []
            for idx in range(error.shape[-1]):
                error_separate.append(error[:,:,idx].mean())
            loss.error_separate = error_separate
            
        return loss
    
    def compute(self, estimates, targets):
        raise NotImplementedError
    

class L2Loss(Loss):
    def compute(self, estimates, targets, activation_function=None):
        # assert estimates.shape == targets.shape, "Input {} and targets {} for L2 loss need to have identical shape!"\
        #     .format(estimates.shape, targets.shape)
        if activation_function is not None:
            estimates = activation_function(estimates)
        if not isinstance(targets, torch.Tensor):
            targets = torch.tensor(targets, device=estimates.device, dtype=estimates.dtype)
        l2_loss = torch.nn.MSELoss(reduction='none')(estimates, targets)
        return l2_loss


class KLDivLoss(Loss):
    def compute(self, estimates, targets):
        if not isinstance(estimates, Gaussian): estimates = Gaussian(estimates)
        if not isinstance(targets, Gaussian): targets = Gaussian(targets)
        kl_divergence = estimates.kl_divergence(targets)
        return kl_divergence


class CELoss(Loss):
    compute = staticmethod(torch.nn.functional.cross_entropy)
    

class PenaltyLoss(Loss):
    def compute(self, val):
        """Computes weighted mean of val as penalty loss."""
        return val


class NLL(Loss):
    # Note that cross entropy is an instance of NLL, as is L2 loss.
    def compute(self, estimates, targets):
        nll = estimates.nll(targets)
        return nll
    

class BCELogitsLoss(Loss):
    def compute(self, estimates, targets):
        return BCEWithLogitsLoss()(estimates, targets)


class MSE(Loss):
    def compute(self, estimates, targets):
        return (estimates - targets)**2
