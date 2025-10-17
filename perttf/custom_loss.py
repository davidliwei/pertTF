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


def CCE_loss(
    input_emb: torch.Tensor,
    input_to_pert_emb: torch.Tensor,
    pert_emb: torch.Tensor,
    pert_to_input_emb: torch.Tensor,
    lambda_fwd: float = 10.0,
    lambda_rev: float = 10.0,
    # Optional arguments for triplet loss on all 4 embeddings
    input_labels: torch.Tensor = None,
    pert_labels: torch.Tensor = None,
    lambda_triplet: float = 1.0,
    triplet_margin: float = 0.5
) -> torch.Tensor:
    """
    Calculates the contrastive losses between cell embedding and perturbed cell embeddings.

    This loss combins up to three components:
    2. Forward Consistency Loss (MSE): Enforces that the predicted perturbed
       embedding is close to the true perturbed embedding.
    3. Reverse Consistency Loss (MSE): Enforces cycle consistency.
    4. Triplet Loss (Optional): Stacks all 4 embeddings and uses labels to structure the space.

    Args:
        decoded_expression (torch.Tensor): The final output from the decoder.
        true_expression (torch.Tensor): The ground truth expression.
        input_emb (torch.Tensor): The embedding of the original input cell.
        pert_to_input_emb (torch.Tensor): The reverse-perturbed embedding.
        input_to_pert_emb (torch.Tensor): The predicted perturbed embedding.
        pert_emb (torch.Tensor): The true embedding of the perturbed cell.
        lambda_fwd (float): Weight for the forward consistency loss.
        lambda_rev (float): Weight for the reverse consistency loss.
        input_labels (torch.Tensor, optional): Labels for the input cells.
        pert_labels (torch.Tensor, optional): Labels for the perturbed cells.
        lambda_triplet (float): Weight for the triplet loss.
        triplet_margin (float): Margin for the triplet loss.

    Returns:
        torch.Tensor: A single scalar value representing the total loss.
    """



    # Base total loss
    total_loss = perturb_embedding_loss(input_emb, input_to_pert_emb, pert_emb, pert_to_input_emb, input_labels, pert_labels, lambda_fwd, lambda_rev)

    # 4. Optional Triplet Loss on stacked embeddings
    if input_labels is not None and pert_labels is not None:
        # Stack all four embeddings into a single large batch
        # New batch size will be 4 * original_batch_size
       # mask = input_labels != pert_labels
       # mask = mask.unsqueeze(1)
       #loss_fwd_consistency = loss_fwd_consistency / F.mse_loss(input_to_pert_emb*mask, input_emb*mask) 
       # loss_rev_consistency = loss_rev_consistency / F.mse_loss(pert_to_input_emb*mask, pert_emb*mask) 

        stacked_embeddings = torch.cat([
            input_emb,
            input_to_pert_emb,
            pert_emb,
            pert_to_input_emb,
            
        ], dim=0)

        # Create corresponding labels for the stacked embeddings
        # input_emb and pert_to_input_emb share the input_labels
        # input_to_pert_emb and pert_emb share the pert_labels
        stacked_labels = torch.cat([
            input_labels,
            pert_labels,
            pert_labels,
            input_labels
            
        ], dim=0)
        
        loss_triplet = all_triplet_loss(stacked_embeddings, stacked_labels, margin=triplet_margin)
        #print(loss_triplet)
        total_loss += lambda_triplet * loss_triplet

    return total_loss



