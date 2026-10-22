import torch

from perttf.model.ontarget import NO_TOKEN, build_pert_gene_token, ontarget_loss

GT2I = {"WT": 0, "A": 1, "B": 2, "C": 3}
STOI = {"<pad>": 0, "A": 1, "B": 2, "Z": 3}


class V:
    stoi = STOI


def test_build_mapping_only_genes_in_both_vocabularies():
    tok = build_pert_gene_token(GT2I, V())
    assert tok.tolist() == [NO_TOKEN, 1, 2, NO_TOKEN]          # WT and C have no gene token
    assert build_pert_gene_token(GT2I, STOI).tolist() == tok.tolist()   # plain dict works too


def test_loss_uses_only_the_on_target_position():
    tok = build_pert_gene_token(GT2I, V())
    gene_ids = torch.tensor([[1, 2, 3], [1, 2, 3]])
    pred = torch.tensor([[1.0, 5.0, 9.0], [1.0, 5.0, 9.0]])
    target = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    loss, n = ontarget_loss(pred, target, gene_ids, torch.tensor([1, 2]), tok)   # genes A, B
    assert n == 2
    assert torch.isclose(loss, torch.tensor((1.0 ** 2 + 5.0 ** 2) / 2))


def test_zero_when_target_absent_or_control():
    tok = build_pert_gene_token(GT2I, V())
    gene_ids = torch.tensor([[3, 3, 3]])
    pred = torch.ones(1, 3, requires_grad=True)
    target = torch.zeros(1, 3)
    for lab in (torch.tensor([1]), torch.tensor([0])):          # gene not in the window / control genotype
        loss, n = ontarget_loss(pred, target, gene_ids, lab, tok)
        assert n == 0 and float(loss.detach()) == 0.0


def test_multi_label_rows_and_padding():
    tok = build_pert_gene_token(GT2I, V())
    gene_ids = torch.tensor([[1, 2, 3]])
    pred = torch.tensor([[2.0, 4.0, 8.0]])
    target = torch.zeros(1, 3)
    loss, n = ontarget_loss(pred, target, gene_ids, torch.tensor([[1, 2, -1]]), tok)
    assert n == 2
    assert torch.isclose(loss, torch.tensor((2.0 ** 2 + 4.0 ** 2) / 2))


def test_gradient_flows_to_the_prediction():
    tok = build_pert_gene_token(GT2I, V())
    pred = torch.zeros(1, 2, requires_grad=True)
    loss, n = ontarget_loss(pred, torch.tensor([[3.0, 0.0]]), torch.tensor([[1, 2]]), torch.tensor([1]), tok)
    loss.backward()
    assert n == 1 and pred.grad[0, 0] != 0 and pred.grad[0, 1] == 0
