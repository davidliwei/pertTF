import torch, pytest
from perttf.model.pertTF import PerturbationTFModel
from perttf.model.hf import HFPerturbationTFModel
from tests.test_pertTF_model import build_model, N_PERT, NTOKENS, D_MODEL

def batch(B=4, L=12, k=10):
    torch.manual_seed(0)
    src = torch.randint(1, NTOKENS, (B, L)); values = torch.rand(B, L)
    mask = torch.zeros(B, L, dtype=torch.bool)
    labels = torch.randint(0, N_PERT, (B, k)); labels[:, 5:] = -1; labels[0, 1:] = -1
    return src, values, mask, labels

@pytest.mark.parametrize("agg", ["sum", "mean", "sqrt"])
def test_forward_width_ten(agg):
    model = build_model(pert_aggregation=agg)
    src, values, mask, labels = batch()
    with torch.no_grad():
        out = model(src, values, mask, pert_labels=labels, pert_labels_next=labels, CLS=True, MVC=True, PERTPRED=True, PSPRED=True)
    assert out["cell_emb_next"].shape == (4, D_MODEL) and torch.isfinite(out["mvc_output_next"]["pred"]).all()

def test_width_one_matches_scalar_forward():
    model = build_model()
    src, values, mask, labels = batch(k=1); labels = labels.clamp_min(0)
    with torch.no_grad():
        a = model(src, values, mask, pert_labels=labels[:, 0], pert_labels_next=labels[:, 0], MVC=True)
        b = model(src, values, mask, pert_labels=labels, pert_labels_next=labels, MVC=True)
    assert torch.allclose(a["cell_emb_next"], b["cell_emb_next"], atol=1e-6)

def test_pert_embeddings_next_3d_is_reduced():
    model = build_model()
    src, values, mask, labels = batch(k=3)
    emb = torch.randn(4, 3, D_MODEL)
    with torch.no_grad():
        out = model(src, values, mask, pert_labels_next=labels, pert_embeddings_next=emb)
    assert out["cell_emb_next"].shape == (4, D_MODEL)

def test_hf_roundtrip_restores_aggregation(tmp_path):
    vocab = {"<pad>": 0, **{f"g{i}": i for i in range(1, NTOKENS)}}
    from perttf.utils.custom_tokenizer import SimpleVocab
    m = HFPerturbationTFModel(vocab=SimpleVocab.from_dict(vocab), genotype_to_index={"WT": 0, "A": 1},
                              cell_type_to_index={"c": 0}, d_model=16, nhead=2, d_hid=16, nlayers=1,
                              n_ps=0, pert_aggregation="mean")
    m.save_pretrained(tmp_path)
    r = HFPerturbationTFModel.from_pretrained(str(tmp_path))
    assert r.pert_aggregation == "mean"
    assert "pert_aggregation" in r._hub_mixin_config
