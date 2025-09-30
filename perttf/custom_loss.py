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

def perturb_embedding_loss(
    input_emb: torch.Tensor,
    input_to_pert_emb: torch.Tensor,
    pert_emb: torch.Tensor,
    pert_to_input_emb: torch.Tensor,
    input_labels: torch.Tensor,
    pert_labels: torch.Tensor,
    lambda_fwd: float = 1.0,
    lambda_rev: float = 1.0
) -> torch.Tensor:
    """
    Calculates the composite loss for the virtual perturbation model.

    This loss combines three components:
    1. Reconstruction Loss (MSE): How well the decoded expression matches the true one.
    2. Forward Consistency Loss (Cosine Distance): Enforces that the predicted perturbed
       embedding is close to the true perturbed embedding.
    3. Reverse Consistency Loss (Cosine Distance): Enforces cycle consistency, ensuring
       the reverse-perturbed embedding is close to the original input embedding.

    Args:
        decoded_expression (torch.Tensor): The final output from the decoder (predicted expression).
        true_expression (torch.Tensor): The ground truth expression of the sampled perturbed cell.
        input_emb (torch.Tensor): The embedding of the original input cell.
        pert_to_input_emb (torch.Tensor): The result of reverse-perturbing the perturbed cell's embedding.
        input_to_pert_emb (torch.Tensor): The result of perturbing the input cell's embedding (the predicted perturbed embedding).
        pert_emb (torch.Tensor): The true embedding of the sampled perturbed cell.
        lambda_fwd (float): The weight for the forward consistency loss.
        lambda_rev (float): The weight for the reverse consistency loss.

    Returns:
        torch.Tensor: A single scalar value representing the total loss.
    """
    #mask = input_labels != pert_labels
    #mask = mask.unsqueeze(1)

    # Forward Consistency Loss (L_fwd_consistency)
    # Cosine distance = 1 - Cosine Similarity.
    # We want to maximize similarity, which is equivalent to minimizing distance.
    # The '.mean()' aggregates the loss across the batch.
    #similarity_fwd = F.Cosine(input_to_pert_emb, pert_emb)
    #loss_fwd_consistency = (1 - similarity_fwd).mean()
    #loss_fwd_consistency = F.relu(F.mse_loss(input_to_pert_emb, pert_emb) -  F.mse_loss(input_to_pert_emb*mask, input_emb*mask) + 0.5)
    loss_fwd_consistency = F.mse_loss(input_to_pert_emb, pert_emb)# -  F.mse_loss(input_to_pert_emb*mask, input_emb*mask) + 0.5)
    #  Reverse Consistency Loss (L_rev_consistency)
    # Similar to the forward loss, this ensures the reverse transformation is valid.
    #similarity_rev = F.mse_loss(pert_to_input_emb, input_emb)
    #loss_rev_consistency = F.relu(F.mse_loss(pert_to_input_emb, input_emb) - F.mse_loss(pert_to_input_emb*mask, pert_emb*mask) + 0.5)
    loss_rev_consistency = F.mse_loss(pert_to_input_emb, input_emb)# - F.mse_loss(pert_to_input_emb*mask, pert_emb*mask) + 0.5)

    # 4. Combine the losses
    # The total loss is a weighted sum of the three components.
    total_loss = lambda_fwd * loss_fwd_consistency + lambda_rev * loss_rev_consistency
    
    # You can optionally return the individual components for monitoring during training
    # return total_loss, loss_recon, loss_fwd_consistency, loss_rev_consistency
    
    return total_loss


def all_triplet_loss(embeddings: torch.Tensor, labels: torch.Tensor, margin: float = 0.5) -> torch.Tensor:
    """
    Calculates the triplet loss for a batch of embeddings using a "batch-all" strategy.

    This method considers all valid anchor-positive-negative triplets within the batch.
    A triplet is valid if the anchor and positive have the same label, and the anchor
    and negative have different labels. The loss is then averaged over all triplets
    that have a positive loss value.

    Args:
        embeddings (torch.Tensor): The batch of embeddings (shape: [batch_size, emb_dim]).
        labels (torch.Tensor): The labels for each embedding (shape: [batch_size]).
        margin (float): The desired margin between positive and negative distances.

    Returns:
        torch.Tensor: A single scalar value for the mean triplet loss.
    """
    # Calculate pairwise squared L2 distances
    pairwise_dist = torch.cdist(embeddings, embeddings, p=2).pow(2)

    # Create masks to identify positive and negative pairs
    mask_positive = (labels.unsqueeze(1) == labels.unsqueeze(0))

    
    mask_positive.fill_diagonal_(False)
    
    mask_negative = ~mask_positive
    mask_negative.fill_diagonal_(False)
    hardest_negative_dist = (pairwise_dist + 1e8 * (~mask_negative)).min(dim=1)[0]
    # --- Batch-All Triplet Mining ---
    # For each anchor, we want to consider all positive and all negative pairs.
    # We can use broadcasting to compute the loss for all possible triplets.
    
    # Reshape distances for broadcasting:
    # anchor_positive_dist[i, j] = distance(i, j)
    # anchor_negative_dist[i, k] = distance(i, k)
    anchor_positive_dist = pairwise_dist.unsqueeze(2)  # Shape: (batch, batch, 1)
    anchor_negative_dist = pairwise_dist.unsqueeze(1)  # Shape: (batch, 1, batch)

    # Calculate the loss for all possible triplets (i, j, k)
    # triplet_loss[i, j, k] = D(i, j) - D(i, k) + margin
    triplet_loss = (anchor_positive_dist - anchor_negative_dist)/hardest_negative_dist.mean() + margin

    # Create a mask for valid triplets. A triplet (i, j, k) is valid if
    # (i, j) is a positive pair and (i, k) is a negative pair.
    mask_valid_triplets = mask_positive.unsqueeze(2) & mask_negative.unsqueeze(1)
    
    # Apply the mask to keep only the loss for valid triplets
    # Set the loss for invalid triplets to 0
    triplet_loss = triplet_loss * mask_valid_triplets
    
    # Remove negative losses (as per the max(0, loss) formulation)
    triplet_loss = F.relu(triplet_loss)

    # Count the number of triplets with positive loss
    num_positive_triplets = (triplet_loss > 1e-16).float().sum()
    
    # Calculate the mean loss over the positive triplets.
    # If there are no positive triplets, the loss is 0.
    if num_positive_triplets > 0:
        loss = triplet_loss.sum() / num_positive_triplets
    else:
        loss = torch.tensor(0.0, device=embeddings.device)

    return loss


def hard_triplet_loss(embeddings: torch.Tensor, labels: torch.Tensor, margin: float = 0.5) -> torch.Tensor:
    """
    Calculates the triplet loss for a batch of embeddings using online hard triplet mining.

    For each anchor in the batch, it finds the hardest positive (most distant sample
    with the same label) and the hardest negative (closest sample with a different
    label) and computes the loss.

    Args:
        embeddings (torch.Tensor): The batch of embeddings (shape: [batch_size, emb_dim]).
        labels (torch.Tensor): The labels for each embedding (shape: [batch_size]).
        margin (float): The desired margin between positive and negative distances.

    Returns:
        torch.Tensor: A single scalar value for the mean triplet loss.
    """
    # Calculate pairwise squared L2 distances
    pairwise_dist = torch.cdist(embeddings, embeddings, p=2).pow(2)

    # Create masks to identify positive and negative pairs
    # mask_positive[i, j] is True if sample i and j have the same label
    mask_positive = (labels.unsqueeze(1) == labels.unsqueeze(0))
    # We need to ignore the distance of a sample to itself (diagonal)
    mask_positive.fill_diagonal_(False)
    
    # mask_negative[i, j] is True if sample i and j have different labels
    mask_negative = ~mask_positive
    mask_negative.fill_diagonal_(False)

    # --- Hard Triplet Mining ---
    # For each anchor, find the hardest positive (max distance)
    # Add a large negative value to non-positive pairs to ensure they aren't chosen
    hardest_positive_dist = (pairwise_dist + -1e8 * (~mask_positive)).max(dim=1)[0]

    # For each anchor, find the hardest negative (min distance)
    # Add a large positive value to non-negative pairs to ensure they aren't chosen
    hardest_negative_dist = (pairwise_dist + 1e8 * (~mask_negative)).min(dim=1)[0]
    
    # Calculate triplet loss for each sample in the batch
    # loss = max(0, D(anchor, positive) - D(anchor, negative) + margin)
    loss = F.relu((hardest_positive_dist - hardest_negative_dist)/ hardest_negative_dist.mean() + margin)

    return loss.mean()









    
def SUPCON_loss(features, labels=None, mask=None, contrast_mode = 'all', temperature = 0.07, base_temperature = 0.5):
    """Compute loss for model. If both `labels` and `mask` are None,
    it degenerates to SimCLR unsupervised loss:
    https://arxiv.org/pdf/2002.05709.pdf

    Args:
        features: hidden vector of shape [bsz, n_views, ...].
        labels: ground truth of shape [bsz].
        mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
            has the same class as sample i. Can be asymmetric.
    Returns:
        A loss scalar.
    """
    device = (torch.device('cuda')
                if features.is_cuda
                else torch.device('cpu'))

    if len(features.shape) < 3:
        raise ValueError('`features` needs to be [bsz, n_views, ...],'
                            'at least 3 dimensions are required')
    if len(features.shape) > 3:
        features = features.view(features.shape[0], features.shape[1], -1)

    batch_size = features.shape[0]
    dim = features.shape[-1]
    if labels is not None and mask is not None:
        raise ValueError('Cannot define both `labels` and `mask`')
    elif labels is None and mask is None:
        mask = torch.eye(batch_size, dtype=torch.float32).to(device)
    elif labels is not None:
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to(device)
    else:
        mask = mask.float().to(device)

    contrast_count = features.shape[1]
    contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
    if contrast_mode == 'one':
        anchor_feature = features[:, 0]
        anchor_count = 1
    elif contrast_mode == 'all':
        anchor_feature = contrast_feature
        anchor_count = contrast_count
    else:
        raise ValueError('Unknown mode: {}'.format(contrast_mode))

    # compute logits
    anchor_dot_contrast = torch.div(
        torch.matmul(anchor_feature, contrast_feature.T),
        temperature*dim)
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    # tile mask
    mask = mask.repeat(anchor_count, contrast_count)
    # mask-out self-contrast cases
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
        0
    )
    mask = mask * logits_mask

    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # compute mean of log-likelihood over positive
    # modified to handle edge cases when there is no positive pair
    # for an anchor point. 
    # Edge case e.g.:- 
    # features of shape: [4,1,...]
    # labels:            [0,1,1,2]
    # loss before mean:  [nan, ..., ..., nan] 
    mask_pos_pairs = mask.sum(1)
    mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

    # loss
    loss = - (temperature / base_temperature) * mean_log_prob_pos
    loss = loss.view(anchor_count, batch_size).mean()

    return loss

