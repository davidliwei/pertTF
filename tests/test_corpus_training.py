"""Two independent on-disk datasets through the actual training/evaluation loop."""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch
import wandb
from scipy import sparse

from scorpus import concat, from_h5ad
from perttf.custom_loss import GenerativeExpressionLoss
from perttf.model.corpus_data import produce_corpus_datasets
from perttf.model.hf import HFPerturbationTFModel
from perttf.model.train_function import evaluate, wrapper_train


def build_corpus(tmp_path):
    members = {}
    for name, genes, perturb_col in (("a", ["g1", "g2", "g3"], "target"), ("b", ["g3", "g1", "g4"], "genotype")):
        obs = pd.DataFrame({
            perturb_col: ["ctrl", "P1", "P1", "ctrl", "P1", "P1"],
            "celltype": ["T"] * 6, "batch": [name] * 6,
        }, index=[f"cell{i}" for i in range(6)])
        counts = np.array([[1, 2, 0], [2, 4, 1], [3, 1, 2], [1, 3, 1], [2, 5, 2], [3, 2, 1]], dtype=np.int32)
        data = ad.AnnData(sparse.csr_matrix(counts), obs=obs, var=pd.DataFrame(index=genes))
        source = tmp_path / f"{name}.h5ad"
        data.write_h5ad(source)
        members[name] = from_h5ad(source, tmp_path / name, chunk_rows=2)
    return concat(
        members,
        obs_columns={
            "perturbation": {"a": "target", "b": "genotype"},
            "celltype": {"a": "celltype", "b": "celltype"},
            "batch": {"a": "batch", "b": "batch"},
        },
        feature_key={"a": "index", "b": "index"},
    )


def test_two_corpora_unified_training_and_validation(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    parameters = dict(
        next_cell_pred_type="pert", GEPC=True, input_emb_style="continuous",
        batch_size=6, max_seq_len=4, mask_ratio=1.0, sampling_mode="uniform",
        seed=4, ps_weight=0.0, pad_token="<pad>", pad_value=-2, mask_value=-1,
        amp=False, dab_weight=0.0, ADV=False, lr=1e-3, schedule_ratio=0.9,
        log_interval=100, use_batch_label=False, perturbation_input=False,
        next_weight=1.0, this_weight=1.0, ecs_thres=0.0, CCE=False,
        explicit_zero_prob=False, distribution=None, mvc_masked_train=False,
        cell_type_classifier=True, genotype_classifier=True,
        cell_type_classifier_weight=0.1, perturbation_classifier_weight=0.1,
        perturbation_metric_min_cells=1, perturbation_metric_mode="mean",
        perturbation_metric_target_sum=10000.0, epochs=1, do_train=True,
        dataset_name="synthetic_corpus",
    )
    with wandb.init(mode="disabled", config=parameters) as run:
        corpus = build_corpus(tmp_path)
        sources = {"features": (["P1", "external_only"], torch.tensor([[1., 0., 2.], [0., 1., 1.]]))}
        tokenizer = {"<pad>": 0, "<cls>": 1, "<eoc>": 2, "g3": 3, "g1": 4, "g2": 5, "g4": 6}
        data_gen = produce_corpus_datasets(
            corpus, run.config,
            train_indices=[0, 1, 2, 6, 7, 8], valid_indices=[3, 4, 5, 9, 10, 11],
            perturbation_column="perturbation", celltype_column="celltype", batch_column="batch",
            control_label="ctrl", pert_sources=sources,
            tokenizer_stoi=tokenizer,
            cell_type_to_index={"unused_context": 0, "T": 1},
            batch_to_index={"b": 0, "a": 1},
        )
        mapping = data_gen["genotype_to_index"]
        assert "external_only" in mapping
        assert data_gen["train_loader"].label_to_index_by_name["perturbation"] == mapping
        assert data_gen["valid_loader"].label_to_index_by_name["perturbation"] == mapping
        assert data_gen["metric_gene_ids"].tolist() == ["g1", "g3"]
        assert data_gen["vocab"].stoi == tokenizer
        assert data_gen["cell_type_to_index"] == {"unused_context": 0, "T": 1}
        assert data_gen["train_loader"].label_to_index_by_name["batch"] == {"b": 0, "a": 1}
        np.testing.assert_array_equal(data_gen["gene_ids"], [4, 5, 3, 6])
        control_counts = np.array([[1., 1.], [3., 1.]])
        treated_counts = np.array([[2., 2.], [3., 1.], [5., 2.], [2., 3.]])
        for label, counts in (("ctrl", control_counts), ("P1", treated_counts)):
            expected_mean = np.log1p(counts / counts.sum(axis=1, keepdims=True) * 10000).mean(axis=0)
            np.testing.assert_allclose(data_gen["perturbation_reference_groups"][("T", label)].mean, expected_mean)
        batch = next(iter(data_gen["valid_loader"]))
        assert set(batch["next_index"].tolist()) == {3, 4, 5, 9, 10, 11}
        source_ds = corpus.take_metadata(batch["index"].numpy(), columns=["dataset_id"])["dataset_id"]
        target_ds = corpus.take_metadata(batch["next_index"].numpy(), columns=["dataset_id"])["dataset_id"]
        assert source_ds == target_ds
        vocab = data_gen["vocab"]
        model = HFPerturbationTFModel(
            n_pert=len(mapping), nlayers_pert=2, n_ps=0, d_model=16, nhead=2, d_hid=32, nlayers=1,
            vocab=vocab, dropout=0.0, pad_token="<pad>", pad_value=-2,
            do_mvc=True, do_dab=False, use_batch_labels=False, num_batch_labels=2,
            domain_spec_batchnorm=False, n_bins=0, ecs_threshold=0.0,
            explicit_zero_prob=False, use_fast_transformer=False, pre_norm=False,
            n_cls=1, nlayers_cls=2, input_emb_style="continuous",
            pert_sources=sources, genotype_to_index=mapping,
            cell_type_to_index=data_gen["cell_type_to_index"], control_label="ctrl",
        )
        before = model.pert_encoder.sources["features"].proj[0].weight.detach().clone()
        save_dir = tmp_path / "training"
        save_dir.mkdir()
        best = wrapper_train(model, run.config, data_gen, save_dir=save_dir, device=torch.device("cpu"))
        assert not torch.equal(before, model.pert_encoder.sources["features"].proj[0].weight)
        assert (save_dir / "best_model.pt").exists()
        saved = torch.load(save_dir / "running_parameters.pt", weights_only=False)
        assert saved["genotype_to_index"] == mapping
        assert saved["pert_source_config"]["control_label"] == "ctrl"
        assert saved["corpus_config"]["adapter"]["normalization_scale"] == 1.0
        result = evaluate(
            best, data_gen["valid_loader"], run.config, vocab, device=torch.device("cpu"),
            cell_type_to_index=data_gen["cell_type_to_index"], genotype_to_index=mapping,
            expr_prediction_mode="mean",
        )
        assert all(np.isfinite(v) for v in result["losses"].values())
        assert set(result["predicted_groups"]) == set(data_gen["perturbation_reference_groups"])
        assert all(group.mean.shape == (2,) for group in result["predicted_groups"].values())
        with pytest.raises(ValueError, match="overlap"):
            produce_corpus_datasets(
                corpus, run.config, train_indices=[0, 1, 2], valid_indices=[0, 1, 4],
                perturbation_column="perturbation", celltype_column="celltype", batch_column="batch",
                control_label="ctrl", pert_sources=sources,
            )


def test_unmeasured_genes_and_empty_mask_do_not_contribute():
    loss = GenerativeExpressionLoss()
    prediction = torch.tensor([[2., 900.]], requires_grad=True)
    output = {"pred": prediction, "distribution": None}
    target = torch.tensor([[1., 0.]])
    actual = loss(output, target, torch.tensor([[True, False]]))
    assert actual.item() == 1.0
    actual.backward()
    assert prediction.grad.tolist() == [[2.0, 0.0]]
    assert loss(output, target, torch.zeros((1, 2), dtype=torch.bool)).item() == 0.0
