# tests/test_multi_pert_inference.py
import numpy as np, pandas as pd, anndata as ad, pytest, torch
from perttf.model.hf import HFPerturbationTFModel
from perttf.utils.custom_tokenizer import SimpleVocab

GENES = [f"g{i}" for i in range(1, 8)]

def model(mapping, **cfg):
    vocab = SimpleVocab.from_dict({"<pad>": 0, "<cls>": 1, **{g: i + 2 for i, g in enumerate(GENES)}})
    return HFPerturbationTFModel(vocab=vocab, genotype_to_index=mapping, cell_type_to_index={"c": 0},
                                 d_model=16, nhead=2, d_hid=16, nlayers=1, n_ps=0, do_mvc=True, training_config=cfg).eval()

def adata(next_labels):
    n = len(next_labels)
    a = ad.AnnData(X=np.ones((n, len(GENES)), dtype=np.float32),
                   obs=pd.DataFrame({"celltype": ["c"] * n, "genotype": ["WT"] * n, "genotype_next": next_labels}),
                   var=pd.DataFrame(index=GENES))
    a.layers["X_binned"] = a.X.copy()
    return a

def test_legacy_checkpoint_with_combo_key_rejected():
    m = model({"WT": 0, "A+B": 1})
    with pytest.raises(ValueError, match="A\\+B"):
        m.predict_perturbations(adata(["WT"]), pert_delimiter="+")

def test_delimiter_mismatch_rejected():
    m = model({"WT": 0, "A": 1}, pert_delimiter="+")
    with pytest.raises(ValueError):
        m.predict_perturbations(adata(["WT"]), pert_delimiter="|")

def test_unknown_component_raises_before_compute():
    m = model({"WT": 0, "A": 1})
    with pytest.raises(ValueError, match="B"):
        m.predict_perturbations(adata(["A+B"]), pert_delimiter="+")

def test_set_inference_runs_and_records_delimiter():
    m = model({"WT": 0, "A": 1, "B": 2}, pert_delimiter="+")
    r = m.predict_perturbations(adata(["A+B", "B", "WT"]), prediction_mode="mean", device="cpu")
    assert r.shape == (3, len(GENES)) and r.uns["perttf_prediction"]["pert_delimiter"] == "+"
    assert r.obs["genotype_next"].tolist() == ["A+B", "B", "WT"]
