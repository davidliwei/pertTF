import math, torch, pytest
from perttf.model.modules import PertLabelEncoder, FeaturePertEncoder
from perttf.model.pert_encoder import UnifiedPertEncoder, encode_perturbation_set, reduce_perturbation_embeddings

D, N = 16, 6
MAPPING = {"WT": 0, "A": 1, "B": 2, "C": 3, "D": 4, "E": 5}

def encoders():
    torch.manual_seed(0)
    yield PertLabelEncoder(N, D)
    yield FeaturePertEncoder(torch.randn(N, 8), D)
    yield UnifiedPertEncoder({"src": (["A", "B"], torch.randn(2, 8).numpy())}, MAPPING, D)

@pytest.mark.parametrize("agg", ["sum", "mean", "sqrt"])
def test_width_one_equals_scalar(agg):
    for enc in encoders():
        enc.eval()
        lab = torch.tensor([1, 2, 0])
        a = encode_perturbation_set(enc, lab, agg)
        b = encode_perturbation_set(enc, lab.unsqueeze(1), agg)
        assert torch.allclose(a, b, atol=1e-6)

def test_pad_contributes_zero_and_sum_is_additive():
    enc = PertLabelEncoder(N, D).eval()
    single = encode_perturbation_set(enc, torch.tensor([1, 2]), "sum")
    pair = encode_perturbation_set(enc, torch.tensor([[1, 2, -1, -1]]), "sum")
    assert torch.allclose(pair[0], single[0] + single[1], atol=1e-6)
    padded = encode_perturbation_set(enc, torch.tensor([[1, -1, -1]]), "sum")
    assert torch.allclose(padded[0], single[0], atol=1e-6)

def test_mean_and_sqrt_scaling():
    enc = PertLabelEncoder(N, D).eval()
    labels = torch.tensor([[1, 2, 3, 4]])
    s = encode_perturbation_set(enc, labels, "sum")
    assert torch.allclose(encode_perturbation_set(enc, labels, "mean"), s / 4, atol=1e-6)
    assert torch.allclose(encode_perturbation_set(enc, labels, "sqrt"), s / math.sqrt(4), atol=1e-6)

def test_unified_never_sees_negative_index():
    enc = list(encoders())[2].eval()
    out = encode_perturbation_set(enc, torch.tensor([[1, 2, -1]]), "sum")
    assert out.shape == (1, D) and torch.isfinite(out).all()

def test_bad_aggregation():
    with pytest.raises(ValueError):
        encode_perturbation_set(PertLabelEncoder(N, D), torch.tensor([[1]]), "max")

def test_reduce_matches_encode():
    enc = PertLabelEncoder(N, D).eval()
    labels = torch.tensor([[1, 2, -1]])
    emb = enc(labels.clamp_min(0))
    assert torch.allclose(reduce_perturbation_embeddings(emb, labels, "sum"),
                          encode_perturbation_set(enc, labels, "sum"), atol=1e-6)
