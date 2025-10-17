# Modified from scGPT
import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional


from scgpt.loss import masked_relative_error


def masked_mse_loss(
    input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None
) -> torch.Tensor:
    """
    Compute the masked MSE loss between input and target.
    """
    if mask is None:
        return F.mse_loss(input, target, reduction="mean")
    mask = mask.float() 
    loss = F.mse_loss(input * mask, target * mask, reduction="sum")
    return loss / mask.sum()


def criterion_neg_log_bernoulli(
    input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None
) -> torch.Tensor:
    """
    Compute the negative log-likelihood of Bernoulli distribution
    """
    if mask is None:
        bernoulli = torch.distributions.Bernoulli(probs=input)
        masked_log_probs = bernoulli.log_prob((target > 0).float())
        return -masked_log_probs.mean()
    mask = mask.float()
    bernoulli = torch.distributions.Bernoulli(probs=input)
    masked_log_probs = bernoulli.log_prob((target > 0).float()) * mask
    return -masked_log_probs.sum() / mask.sum()

