"""Sparsely updated perturbation tables must not be frozen by the amp eps (1e-4) of the shared Adam/AdamW optimizer.

Densely updated parameters -- including the gene-token / value-bin / batch-label embeddings -- must keep that eps,
so the fix does not change the training of anything but the perturbation tables.
"""
import torch
import pytest

from perttf.model.modules import PertLabelEncoder
from perttf.utils.set_optimizer import create_optimizer_dict


class _Config(dict):
    """Minimal stand-in for the wandb config object (attribute + .get access)."""
    __getattr__ = dict.__getitem__


def _config(**kw):
    base = dict(amp=True, dab_weight=0.0, ADV=False, lr=1e-3, optimizer="adamw", scheduler="step",
                schedule_ratio=0.97, epochs=1)
    base.update(kw)
    return _Config(base)


class _Model(torch.nn.Module):
    """A sparsely updated perturbation table next to a densely updated token embedding."""

    def __init__(self, n_pert=64, n_token=32, d=8):
        super().__init__()
        self.pert_encoder = PertLabelEncoder(n_pert, d)
        self.token_embedding = torch.nn.Embedding(n_token, d)
        self.linear = torch.nn.Linear(d, 1)

    def forward(self, pert_idx, scale=1.0):
        tokens = self.token_embedding(torch.arange(self.token_embedding.num_embeddings)).mean(0)
        return scale * self.linear(self.pert_encoder(pert_idx) + tokens).sum()


def _group_of(optimizer, parameter):
    return [g for g in optimizer.param_groups if any(p is parameter for p in g["params"])][0]


def _optimizer(model, **cfg):
    return create_optimizer_dict(model, torch.device("cpu"), _config(**cfg))


@pytest.mark.parametrize("optimizer_name", ["adam", "adamw"])
def test_perturbation_table_gets_a_small_eps_under_amp(optimizer_name):
    model = _Model()
    d = _optimizer(model, optimizer=optimizer_name)
    assert _group_of(d["optimizer"], model.pert_encoder.embedding.weight)["eps"] == 1e-8
    assert d["optimizer_config"]["sparse_embedding_eps"] == 1e-8


@pytest.mark.parametrize("optimizer_name", ["adam", "adamw"])
def test_dense_parameters_keep_the_amp_eps(optimizer_name):
    """The regression guard for the fix itself: it must not touch anything but the perturbation tables."""
    model = _Model()
    optimizer = _optimizer(model, optimizer=optimizer_name)["optimizer"]
    for parameter in (model.token_embedding.weight, model.linear.weight, model.pert_encoder.enc_norm.weight):
        assert _group_of(optimizer, parameter)["eps"] == 1e-4


def test_sparse_embedding_eps_is_configurable():
    model = _Model()
    optimizer = _optimizer(model, sparse_embedding_eps=1e-6)["optimizer"]
    assert _group_of(optimizer, model.pert_encoder.embedding.weight)["eps"] == 1e-6


def test_without_amp_every_group_uses_1e_8():
    optimizer = _optimizer(_Model(), amp=False)["optimizer"]
    assert all(g["eps"] == 1e-8 for g in optimizer.param_groups)


def test_a_model_without_a_sparse_table_gets_a_single_group():
    class Plain(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(8, 4)

    d = _optimizer(Plain())
    assert len(d["optimizer"].param_groups) == 1
    assert "sparse_embedding_eps" not in d["optimizer_config"]


def test_a_sparsely_updated_row_actually_moves():
    """The bug. A perturbation row is indexed by few samples of few batches, so its gradient elements are routinely
    well below 1e-4; with the amp eps the step lr * m / (sqrt(v) + eps) is ~0 and the row stays put. `scale` puts the
    toy model in that gradient regime."""
    torch.manual_seed(0)
    model = _Model()
    optimizer = _optimizer(model, weight_decay=0.0)["optimizer"]
    row = 3
    before = model.pert_encoder.embedding.weight[row].detach().clone()
    for _ in range(5):                                  # only `row` is ever indexed -> a sparse gradient
        optimizer.zero_grad()
        model(torch.tensor([row]), scale=1e-4).backward()
        assert model.pert_encoder.embedding.weight.grad[row].abs().max() < 1e-4   # the regime the amp eps swamps
        optimizer.step()
    moved = (model.pert_encoder.embedding.weight[row] - before).norm() / before.norm()
    # 5 steps of lr 1e-3 on a unit-ish row: ~5e-3 when the step is sign-like, ~0 when eps dominates.
    assert moved > 1e-3, f"perturbation row barely moved ({moved:.2e}) -- frozen by the optimizer eps"
