"""cell_emb_norm: optional normalisation of the cell side of the next-cell concatenation.

The perturbation encoder ends in a LayerNorm, so the gene side reaches pert_exp_encoder at ~unit scale per
dimension while cell_emb enters raw; measurements on trained pancreatic models show the next-cell prediction then
varies far less across cells than across genes. These tests pin the three settings down:
  - the default ('none') adds no parameters and leaves the forward output bit-identical to the old code path;
  - 'layernorm' / 'batchnorm' insert the module and change cell_emb_next;
  - 'batchnorm' hands the first layer ~unit per-dimension variation ACROSS cells, which raw cell_emb does not.
"""
import pytest
import torch
from torch import nn

from perttf.model.hf import HFPerturbationTFModel
from perttf.utils.custom_tokenizer import SimpleVocab

GENES = ["<pad>", "<cls>", "<eoc>", "A", "B", "C", "D"]
VOCAB = SimpleVocab.from_dict({g: i for i, g in enumerate(GENES)})


def build(kind, seed=0):
    torch.manual_seed(seed)
    kw = {} if kind is None else {"cell_emb_norm": kind}
    return HFPerturbationTFModel(
        vocab=VOCAB, genotype_to_index={"WT": 0, "A": 1}, cell_type_to_index={"c": 0},
        d_model=8, nhead=2, d_hid=8, nlayers=1, nlayers_cls=2, nlayers_pert=2, n_pert=2, n_cls=1, n_ps=1,
        dropout=0.0, do_mvc=True, pert_dim=4, **kw).eval()


def forward(model, n=6):
    torch.manual_seed(1)
    src = torch.randint(3, len(GENES), (n, 4))
    values = torch.randn(n, 4)
    return model(src, values, src_key_padding_mask=torch.zeros(n, 4, dtype=torch.bool),
                 pert_labels_next=torch.zeros(n, dtype=torch.long), CLS=False, MVC=False)


def test_default_is_identity_and_adds_no_parameters():
    m0, m1 = build(None), build("none")
    assert isinstance(m1.cell_emb_norm, nn.Identity)
    assert set(m0.state_dict()) == set(m1.state_dict())
    assert not any(k.startswith("cell_emb_norm") for k in m1.state_dict())
    with torch.no_grad():
        a, b = forward(m0)["cell_emb_next"], forward(m1)["cell_emb_next"]
    assert torch.equal(a, b)


@pytest.mark.parametrize("kind,cls", [("layernorm", nn.LayerNorm), ("batchnorm", nn.BatchNorm1d)])
def test_norm_is_inserted(kind, cls):
    m = build(kind)
    assert isinstance(m.cell_emb_norm, cls)
    assert any(k.startswith("cell_emb_norm") for k in m.state_dict())
    with torch.no_grad():
        assert forward(m)["cell_emb_next"].shape == forward(build("none"))["cell_emb_next"].shape


def test_layernorm_is_idempotent_on_an_already_standardised_cls_embedding():
    """cell_emb is the transformer's CLS output, which a post-LN transformer has already LayerNormed, so a second
    LayerNorm leaves it unchanged (exactly so here, where the transformer's own LN still has its unit init). This
    is why the cell/gene imbalance is NOT a magnitude mismatch that LayerNorm could fix: cell_emb already arrives
    at ~unit scale per dimension; what is small is its variation ACROSS cells."""
    with torch.no_grad():
        ref, got = forward(build("none"))["cell_emb_next"], forward(build("layernorm"))["cell_emb_next"]
    assert torch.allclose(got, ref, atol=1e-5)


def test_batchnorm_changes_the_next_embedding():
    with torch.no_grad():
        ref, got = forward(build("none"))["cell_emb_next"], forward(build("batchnorm").train())["cell_emb_next"]
    assert not torch.allclose(got, ref)


def test_batchnorm_equalises_the_per_dimension_spread_across_cells():
    m = build("batchnorm").train()
    torch.manual_seed(2)
    # a cell_emb-like batch: a large shared mean with small across-cell deviations (what the trained models show)
    C = torch.ones(16, 8) * 3.0 + 0.02 * torch.randn(16, 8)
    assert C.std(0).mean() < 0.05
    with torch.no_grad():
        out = m.cell_emb_norm(C)
    assert out.std(0).mean() == pytest.approx(1.0, abs=0.15)


def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        build("zscore")
