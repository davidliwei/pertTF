import torch
import torch.nn.functional as F


def _sample_zig(mu, theta, pi, sample=False, to_counts=False, sf=None):
    """Zero-Inflated Gaussian (Log-Space). Theta = Sigma."""
    if not sample:
        val = (1 - pi) * mu
    else:
        is_zero = torch.bernoulli(pi).bool()
        epsilon = torch.randn_like(mu)
        val = mu + theta * epsilon  # theta is sigma here
        val = val.masked_fill(is_zero, 0.0)
    
    if to_counts:
        val = torch.expm1(val)
        if sf is not None:
            if val.dim() > sf.dim():
                sf = sf.unsqueeze(-1)
            val = val / (sf + 1e-8)
    return val

def _sample_nb(mu, theta, pi, sample=False, **kwargs):
    """Negative Binomial. Theta = Dispersion."""
    eps = 1e-6
    nb_logits = (mu + eps).log() - (theta + eps).log()
    dist = torch.distributions.NegativeBinomial(total_count=theta, logits=nb_logits)
    return dist.sample() if sample else mu

def _sample_zinb(mu, theta, pi, sample=False, **kwargs):
    """Zero-Inflated NB."""
    val = _sample_nb(mu, theta, pi, sample)
    if sample:
        is_dropout = torch.bernoulli(pi).bool()
        val = val.masked_fill(is_dropout, 0.0)
    else:
        val = (1 - pi) * val
    return val

def _sample_hnb(mu, theta, pi, sample=False, **kwargs):
    """Hurdle NB (Rejection Sampling)."""
    if not sample:
        return (1 - pi) * mu
    
    is_zero = torch.bernoulli(pi).bool()
    
    # Base NB Distribution
    eps = 1e-6
    nb_logits = (mu + eps).log() - (theta + eps).log()
    dist = torch.distributions.NegativeBinomial(total_count=theta, logits=nb_logits)
    
    samples = dist.sample()
    
    # Rejection Loop for "Expressed" slots
    mask_redraw = (samples == 0) & (~is_zero)
    max_iter = 100
    i = 0
    while mask_redraw.any() and i < max_iter:
        new_samples = dist.sample()
        samples = torch.where(mask_redraw, new_samples, samples)
        mask_redraw = (samples == 0) & (~is_zero)
        i += 1
        
    samples = torch.where(mask_redraw, torch.ones_like(samples), samples)
    samples = samples.masked_fill(is_zero, 0.0)
    return samples

import torch

def _sample_hnb_optimized(mu, theta, pi, sample=False, threshold=0.95, **kwargs):
    """
    Optimized Hurdle NB Sampling.
    Uses an analytical approximation for low-mean (high-dropout) genes 
    to prevent the rejection loop from stalling.
    """
    if not sample:
        return (1 - pi) * mu
    
    # 1. Binary Gate (Bernoulli)
    # If pi=1 (dropout), we get 0. If pi=0, we get >0.
    is_zero = torch.bernoulli(pi).bool()
    
    # 2. Prepare NB Distribution
    eps = 1e-6
    # Logits are numerically more stable for low mu
    nb_logits = (mu + eps).log() - (theta + eps).log()
    dist = torch.distributions.NegativeBinomial(total_count=theta, logits=nb_logits)
    
    # 3. Calculate P(Zero) analytically
    # P(0) = (theta / (theta + mu))^theta
    # Using log-space for stability: theta * (log(theta) - log(theta+mu))
    log_p0 = theta * (torch.log(theta + eps) - torch.log(theta + mu + eps))
    p0 = torch.exp(log_p0)

    # 4. Hybrid Initialization
    # "Stubborn" pixels are those where P(0) is very high (e.g. > 95%)
    # For these, the conditional distribution P(x|x>0) is approx 100% at x=1.
    # We force them to 1.0 to skip the infinite loop.
    is_stubborn = p0 > threshold
    
    # Initial Sample
    samples = dist.sample()
    
    # Force stubborn cases to 1.0 (Approximation)
    # This prevents them from triggering the redraw loop
    samples = torch.where(is_stubborn & (samples == 0), torch.tensor(1.0, device=mu.device), samples)

    # 5. Optimized Rejection Loop
    # We only care about (samples == 0) that are NOT stubborn and NOT supposed to be zero
    mask_redraw = (samples == 0) & (~is_zero) & (~is_stubborn)
    
    # The loop will now exit very quickly because we removed the "hard" cases
    max_iter = 100 # Safety net
    i = 0
    
    while mask_redraw.any() and i < max_iter:
        # Resample everything (fastest on GPU even if wasteful)
        new_samples = dist.sample()
        
        # Only update the indices that need it
        samples = torch.where(mask_redraw, new_samples, samples)
        
        # Update mask
        mask_redraw = (samples == 0) & (~is_zero) & (~is_stubborn)
        i += 1

    # 6. Final Cleanup
    # Apply the hurdle (zeros from step 1)
    samples = samples.masked_fill(is_zero, 0.0)
    
    # Failsafe for any stubborn points that slipped through or max_iter hits
    # If it's still 0 but shouldn't be, force to 1
    final_fix_mask = (samples == 0) & (~is_zero)
    samples = samples.masked_fill(final_fix_mask, 1.0)
    
    return samples

def _sample_pois(mu, theta, pi, sample=False, **kwargs):
    """Poisson."""
    dist = torch.distributions.Poisson(rate=mu)
    return dist.sample() if sample else mu

def _sample_zipois(mu, theta, pi, sample=False, **kwargs):
    """Zero-Inflated Poisson."""
    val = _sample_pois(mu, theta, pi, sample)
    if sample:
        is_dropout = torch.bernoulli(pi).bool()
        val = val.masked_fill(is_dropout, 0.0)
    else:
        val = (1 - pi) * val
    return val

# --- 2. Generation Registry ---

GENERATION_REGISTRY = {
    'zig': _sample_zig,
    'nb': _sample_nb,
    'zinb': _sample_zinb,
    'hnb': _sample_hnb_optimized,
    'pois': _sample_pois,
    'zipois': _sample_zipois
}

class DistributionGenerator:
    """
    Wrapper class that dispatches to the registry.
    """
    def __init__(self, distribution: str = None):
        # not sure what we should put here for now
        self.distribution = distribution
        if self.distribution is not None:
            assert type(self.distribution) == str, f"Distribution '{self.distribution}' not of type String."
            self.distribution = self.distribution.lower()
            if self.distribution and distribution not in GENERATION_REGISTRY:
                raise ValueError(f"Distribution '{self.distribution}' not found in registry.")
    
    def generate(
        self,
        outputs: dict,
        n_samples: int = 1,
        sample: bool = False,
        to_counts: bool = False,
        target_size_factor: torch.Tensor = None,
        to_numpy = False,
        device = 'cuda'
    ) -> dict:
        """
        Generate function for expression prediction outputs:
        Args:
            outputs: dictionary: {'pred': non-zero distribution mean, 'param2': dispersion/sigma, 'zero_probs': its should be NON-zero probability}
            n_samples: number of samples to sample for each input sample
            sample: whether to sample or just give mean of distribution
            to_counts: when using 'zig' should convert to counts
            target_size_factor: when using 'zig' should convert to counts scaled by size factor
            to_numpy: output numpy arrays
        Returns:
            dictionary: {'pred', 'param2', 'zero_probs'}
        """
        mu = torch.as_tensor(outputs['pred']).to(device)
        param2 = torch.as_tensor(outputs['param2']).to(device)
        pi = 1.0 - torch.as_tensor(outputs['zero_probs']).to(device) # Convert NonZero Prob -> Dropout Prob
        if self.distribution is None:
            return {
                'pred': mu.cpu().numpy() if to_numpy else mu,
                'param2': param2.cpu().numpy() if to_numpy else param2,
                'zero_probs': pi.cpu().numpy() if to_numpy else pi
            }
        
        # Expand for N samples
        if n_samples > 1:
            mu = mu.unsqueeze(-1).expand(-1, -1, n_samples)
            pi = pi.unsqueeze(-1).expand(-1, -1, n_samples)
            if param2 is not None:
                param2 = param2.unsqueeze(-1).expand(-1, -1, n_samples)

        # Dispatch
        generator_func = GENERATION_REGISTRY[self.distribution]
        
        val = generator_func(
            mu=mu, 
            theta=param2, 
            pi=pi, 
            sample=sample, 
            to_counts=to_counts, 
            sf=target_size_factor
        )

        return {
            'pred': val.cpu().numpy() if to_numpy else val,
            'param2': param2.cpu().numpy() if to_numpy and param2 is not None else param2,
            'zero_probs': outputs['zero_probs'].cpu().numpy() if to_numpy and outputs['zero_probs'] is not None else outputs['zero_probs']
        }