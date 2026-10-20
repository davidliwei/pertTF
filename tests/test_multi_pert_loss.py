import torch, pytest, torch.nn.functional as F
from perttf.custom_loss import perturbation_classification_loss, multi_hot_targets, perturbation_overlap_mask, cce_loss

def test_ce_for_scalar_labels():
    logits = torch.randn(3, 5); labels = torch.tensor([0, 4, 2])
    assert torch.allclose(perturbation_classification_loss(logits, labels), F.cross_entropy(logits, labels))

def test_multi_hot_ignores_pad():
    t = multi_hot_targets(torch.tensor([[1, 3, -1], [0, -1, -1]]), 5)
    assert t.tolist() == [[0, 1, 0, 1, 0], [1, 0, 0, 0, 0]]

def test_bce_for_2d_labels():
    logits = torch.randn(2, 5); labels = torch.tensor([[1, 3, -1], [0, -1, -1]])
    expected = F.binary_cross_entropy_with_logits(logits, multi_hot_targets(labels, 5))
    assert torch.allclose(perturbation_classification_loss(logits, labels), expected)
    assert perturbation_classification_loss(logits, labels, pos_weight=3.0) != expected

def test_overlap_mask():
    a = torch.tensor([[1, 2, -1], [3, -1, -1]]); b = torch.tensor([[2, -1, -1], [0, 4, -1], [3, 1, -1]])
    assert perturbation_overlap_mask(a, b, 5).tolist() == [[1, 0, 1], [0, 0, 1]]

def test_cce_loss_accepts_mask():
    d = {"orig_emb0": torch.randn(4, 8), "next_emb0": torch.randn(4, 8)}
    labels = torch.tensor([[1, 2, -1], [2, -1, -1], [3, -1, -1], [0, -1, -1]])
    both = torch.cat([labels, labels])
    mask = perturbation_overlap_mask(both, both, 5)
    loss = cce_loss(d, None, None, pert_mask=mask)
    assert torch.isfinite(loss)
