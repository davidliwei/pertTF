"""On-target supervision for the next-cell prediction.

`mvc_next` supervises the predicted expression of the paired perturbed cell over every sampled gene token, so the
perturbed gene itself carries a weight of ~1/seq_len. Diagnostics on two datasets (pancreatic DE/PP and Jurkat
high-MOI) showed the resulting models predict no gene-specific on-target change: the decoded change of gene g under
`genotype_next = g` is indistinguishable from its change under an unrelated perturbation. `ontarget_loss` adds an
explicit term on exactly those positions, so the model is pushed to represent "perturbing g changes g".

Usage (training loop):
    model.register_buffer("pert_gene_token", build_pert_gene_token(genotype_to_index, vocab), persistent=False)
    ...
    loss_ot, n = ontarget_loss(mvc_output_next["pred"], target_values_next, input_gene_ids,
                               perturbation_labels_next, model.pert_gene_token)
    loss = loss + config.ontarget_weight * loss_ot
"""
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

NO_TOKEN = -1


def build_pert_gene_token(genotype_to_index: Dict[str, int], vocab, n_pert: Optional[int] = None) -> Tensor:
    """genotype index -> gene token id of the SAME gene, or NO_TOKEN when the gene is not in the gene vocabulary.

    `vocab` is anything with a `stoi` dict or a plain {gene: token} mapping (e.g. a scGPT GeneVocab).
    Control / non-gene genotype names (e.g. 'WT') simply have no matching token and stay NO_TOKEN.
    """
    stoi = getattr(vocab, "stoi", vocab)
    size = int(n_pert if n_pert is not None else (max(genotype_to_index.values()) + 1))
    tok = torch.full((size,), NO_TOKEN, dtype=torch.long)
    for gene, idx in genotype_to_index.items():
        i = int(idx)
        if 0 <= i < size and gene in stoi:
            tok[i] = int(stoi[gene])
    return tok


def ontarget_loss(pred: Tensor, target: Tensor, gene_ids: Tensor, pert_labels: Tensor,
                  pert_gene_token: Tensor) -> Tuple[Tensor, int]:
    """Mean squared error on the perturbed gene's own position in the next-cell prediction.

    pred / target: (batch, seq_len) predicted / observed expression of the next cell on the sampled gene tokens.
    gene_ids: (batch, seq_len) the sampled gene tokens.
    pert_labels: (batch,) or (batch, k) next-cell genotype indices (k = multi-perturbation width, padding < 0).
    Returns (loss, number of contributing positions); the loss is a zero scalar (no grad) when nothing contributes.
    """
    if pert_labels.dim() == 1:
        pert_labels = pert_labels.unsqueeze(1)
    tok = pert_gene_token.to(gene_ids.device)
    lab = pert_labels.to(gene_ids.device).long()
    valid = (lab >= 0) & (lab < tok.numel())
    gt = torch.where(valid, tok[lab.clamp(min=0, max=max(tok.numel() - 1, 0))], torch.full_like(lab, NO_TOKEN))
    # (batch, seq_len): a position contributes if its gene token is the on-target gene of any of the cell's labels
    hit = (gene_ids.unsqueeze(2) == gt.unsqueeze(1)) & (gt.unsqueeze(1) != NO_TOKEN)
    hit = hit.any(dim=2)
    n = int(hit.sum().item())
    if n == 0:
        return pred.sum() * 0.0, 0
    return ((pred - target) ** 2)[hit].mean(), n
