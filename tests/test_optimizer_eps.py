"""Embeddings must not be frozen by the amp eps (1e-4) of the shared Adam/AdamW optimizer."""
import torch
import pytest

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
    def __init__(self, n=64, d=8):
        super().__init__()
        self.embedding = torch.nn.Embedding(n, d)
        self.linear = torch.nn.Linear(d, 1)

    def forward(self, idx, scale=1.0):
        return scale * self.linear(self.embedding(idx)).sum()


def _groups(optimizer, model):
    emb = {id(p) for p in model.embedding.parameters()}
    return ([g for g in optimizer.param_groups if any(id(p) in emb for p in g["params"])][0],
            [g for g in optimizer.param_groups if not any(id(p) in emb for p in g["params"])][0])


@pytest.mark.parametrize("optimizer_name", ["adam", "adamw"])
def test_embedding_group_keeps_a_small_eps_under_amp(optimizer_name):
    model = _Model()
    d = create_optimizer_dict(model, torch.device("cpu"), _config(optimizer=optimizer_name))
    emb_group, other_group = _groups(d["optimizer"], model)
    assert emb_group["eps"] == 1e-8
    assert other_group["eps"] == 1e-4          # unchanged amp default for the dense parameters
    assert d["optimizer_config"]["embedding_eps"] == 1e-8


def test_embedding_eps_is_configurable():
    model = _Model()
    d = create_optimizer_dict(model, torch.device("cpu"), _config(embedding_eps=1e-6))
    assert _groups(d["optimizer"], model)[0]["eps"] == 1e-6


def test_without_amp_every_group_uses_1e_8():
    model = _Model()
    d = create_optimizer_dict(model, torch.device("cpu"), _config(amp=False))
    assert all(g["eps"] == 1e-8 for g in d["optimizer"].param_groups)


def test_a_sparsely_updated_embedding_row_actually_moves():
    """The regression. A row of a large table is indexed by few samples of few batches, so its gradient elements are
    routinely well below 1e-4; with the amp eps on the embedding group the step lr * m / (sqrt(v) + eps) is ~0 and the
    row stays put. `scale` reproduces that regime (gradient ~1e-5) in a toy model."""
    torch.manual_seed(0)
    model = _Model()
    optimizer = create_optimizer_dict(model, torch.device("cpu"), _config(weight_decay=0.0))["optimizer"]
    row = 3
    before = model.embedding.weight[row].detach().clone()
    for _ in range(5):                          # only `row` is ever indexed -> a sparse gradient
        optimizer.zero_grad()
        model(torch.tensor([row]), scale=1e-4).backward()
        assert model.embedding.weight.grad[row].abs().max() < 1e-4     # the regime the amp eps swamps
        optimizer.step()
    moved = (model.embedding.weight[row] - before).norm() / before.norm()
    # 5 steps of lr 1e-3 on a unit-ish row: ~5e-3 when the step is sign-like, ~0 when eps dominates.
    assert moved > 1e-3, f"embedding row barely moved ({moved:.2e}) -- frozen by the optimizer eps"
