import numpy as np
import polars as pl
import pytest

from perturb_data_lab.loaders import MetadataIndex
from perttf.model.corpus_adapter import (
    PertTFAdapterConfig,
    PerturbationPairSampler,
)
from perttf.model.corpus_adapter import (
    _positions_for_global_rows,
    _prepare_perttf_metadata,
)


def _build_metadata_index() -> MetadataIndex:
    return MetadataIndex(
        pl.DataFrame(
            {
                "global_row_index": np.arange(8, dtype=np.int64),
                "cell_id": [f"cell_{idx}" for idx in range(8)],
                "dataset_id": ["ds0", "ds0", "ds0", "ds0", "ds0", "ds1", "ds1", "ds1"],
                "dataset_index": np.asarray([0, 0, 0, 0, 0, 1, 1, 1], dtype=np.int32),
                "local_row_index": np.arange(8, dtype=np.int64),
                "size_factor": np.asarray([1.0, 1.1, 1.2, 0.9, 1.0, 1.3, 1.4, 1.5], dtype=np.float32),
                "cell_context": ["T_cell", "T_cell", "T_cell", "B_cell", "B_cell", "T_cell", "T_cell", "B_cell"],
                "perturb_label": ["WT", "KO_A", "KO_B", "WT", "KO_C", "WT", "KO_A", "KO_D"],
                "batch_id": ["b0", "b1", "b1", "b2", "b2", "b3", "b4", "b5"],
            }
        )
    )


def _build_two_row_metadata_index() -> MetadataIndex:
    return MetadataIndex(
        pl.DataFrame(
            {
                "global_row_index": np.arange(2, dtype=np.int64),
                "cell_id": ["cell_0", "cell_1"],
                "dataset_id": ["ds0", "ds0"],
                "dataset_index": np.asarray([0, 0], dtype=np.int32),
                "local_row_index": np.arange(2, dtype=np.int64),
                "size_factor": np.asarray([1.0, 1.1], dtype=np.float32),
                "cell_context": ["T_cell", "T_cell"],
                "perturb_label": ["WT", "KO_A"],
                "batch_id": ["b0", "b1"],
            }
        )
    )


def _build_null_context_metadata_index() -> MetadataIndex:
    return MetadataIndex(
        pl.DataFrame(
            {
                "global_row_index": np.arange(2, dtype=np.int64),
                "cell_id": ["cell_0", "cell_1"],
                "dataset_id": ["ds0", "ds0"],
                "dataset_index": np.asarray([0, 0], dtype=np.int32),
                "local_row_index": np.arange(2, dtype=np.int64),
                "size_factor": np.asarray([1.0, 1.1], dtype=np.float32),
                "cell_context": [None, None],
                "perturb_label": ["WT", "KO_A"],
                "batch_id": ["b0", "b1"],
            }
        )
    )


def _default_config(**kwargs) -> PertTFAdapterConfig:
    return PertTFAdapterConfig(control_labels=("WT",), **kwargs)


def _dataset_group_config(*group_labels: str, **kwargs) -> PertTFAdapterConfig:
    return PertTFAdapterConfig(
        control_labels=("WT",),
        label_fields={
            "perturb_label": "perturbation",
            "cell_context": "celltype",
            "batch_id": "batch",
            "dataset_id": "dataset",
        },
        pairing_group_labels=group_labels,
        **kwargs,
    )


def _build_sampler(
    metadata_index: MetadataIndex,
    *,
    batch_size: int,
    config: PertTFAdapterConfig,
    row_indices=None,
    source_indices=None,
    target_candidate_indices=None,
    seed: int = 0,
    drop_last: bool = True,
    perturbed_target_policy: str = "self_to_control_label",
    pairing_mode: str = "source_driven",
) -> PerturbationPairSampler:
    prepared = _prepare_perttf_metadata(
        metadata_index,
        config=config,
        row_indices=row_indices,
        source_indices=source_indices,
        target_candidate_indices=target_candidate_indices,
    )
    return PerturbationPairSampler(
        prepared.frame,
        batch_size=batch_size,
        perturbation_column=f"{config.perturbation_label}_id",
        control_perturbation_ids=prepared.control_label_ids,
        pairing_group_columns=tuple(f"{label_name}_id" for label_name in config.pairing_group_labels),
        source_positions=prepared.row_selection.source_positions,
        target_candidate_positions=prepared.row_selection.target_candidate_positions,
        global_positions="global_row_index",
        seed=seed,
        drop_last=drop_last,
        perturbed_target_policy=perturbed_target_policy,
        pairing_mode=pairing_mode,
    )


def test_default_pairing_can_cross_dataset_and_context_when_unconstrained() -> None:
    sampler = _build_sampler(
        _build_metadata_index(),
        batch_size=1,
        config=_default_config(),
        source_indices=[0],
        target_candidate_indices=[7],
        seed=7,
        drop_last=False,
    )

    batch = sampler.pair_source_positions([0], seed=11)

    np.testing.assert_array_equal(batch.source_indices, np.asarray([0], dtype=np.int64))
    np.testing.assert_array_equal(batch.target_indices, np.asarray([7], dtype=np.int64))
    np.testing.assert_array_equal(batch.target_perturbation_ids, np.asarray([4], dtype=np.int64))


def test_explicit_dataset_pairing_uses_configured_dataset_label() -> None:
    sampler = _build_sampler(
        _build_metadata_index(),
        batch_size=1,
        config=_dataset_group_config("dataset"),
        source_indices=[0],
        target_candidate_indices=[1, 6],
        seed=7,
        drop_last=False,
    )

    batch = sampler.pair_source_positions([0], seed=11)

    np.testing.assert_array_equal(batch.target_indices, np.asarray([1], dtype=np.int64))


def test_explicit_dataset_and_celltype_pairing_uses_both_group_labels() -> None:
    sampler = _build_sampler(
        _build_metadata_index(),
        batch_size=1,
        config=_dataset_group_config("dataset", "celltype"),
        source_indices=[3],
        target_candidate_indices=[4, 7],
        seed=7,
        drop_last=False,
    )

    batch = sampler.pair_source_positions([3], seed=11)

    np.testing.assert_array_equal(batch.target_indices, np.asarray([4], dtype=np.int64))


def test_perturbed_sources_default_to_self_with_control_label_override() -> None:
    sampler = _build_sampler(
        _build_metadata_index(),
        batch_size=4,
        config=_default_config(),
        seed=3,
    )

    batch = sampler.pair_source_positions([1, 6])

    np.testing.assert_array_equal(batch.target_indices, np.asarray([1, 6], dtype=np.int64))
    np.testing.assert_array_equal(batch.target_perturbation_ids, np.asarray([0, 0], dtype=np.int64))


def test_perturbed_sources_can_pair_to_self_with_source_label() -> None:
    sampler = _build_sampler(
        _build_metadata_index(),
        batch_size=4,
        config=_default_config(),
        perturbed_target_policy="self_to_self_label",
        seed=3,
    )

    batch = sampler.pair_source_positions([1, 2, 4])

    np.testing.assert_array_equal(batch.target_indices, np.asarray([1, 2, 4], dtype=np.int64))
    np.testing.assert_array_equal(batch.target_perturbation_ids, np.asarray([1, 2, 3], dtype=np.int64))


def test_control_source_without_treated_pool_can_pair_to_self() -> None:
    sampler = _build_sampler(
        _build_metadata_index(),
        batch_size=1,
        config=_dataset_group_config("dataset", "celltype"),
        row_indices=[3],
        perturbed_target_policy="self_to_self_label",
        drop_last=False,
    )

    batch = sampler.pair_source_positions([0])

    np.testing.assert_array_equal(batch.target_indices, np.asarray([3], dtype=np.int64))
    np.testing.assert_array_equal(batch.target_perturbation_ids, np.asarray([0], dtype=np.int64))


def test_matched_control_policy_respects_explicit_pairing_groups() -> None:
    sampler = _build_sampler(
        _build_metadata_index(),
        batch_size=4,
        config=_dataset_group_config("dataset", "celltype"),
        perturbed_target_policy="matched_control_cell",
        seed=13,
    )

    batch = sampler.pair_source_positions([1, 2, 6], seed=19)

    np.testing.assert_array_equal(batch.target_indices, np.asarray([0, 0, 5], dtype=np.int64))
    np.testing.assert_array_equal(batch.target_perturbation_ids, np.asarray([0, 0, 0], dtype=np.int64))


def test_target_driven_pairing_visits_each_target_with_matched_control() -> None:
    sampler = _build_sampler(
        _build_metadata_index(),
        batch_size=2,
        config=_dataset_group_config("dataset", "celltype"),
        source_indices=[0, 3, 5],
        target_candidate_indices=[1, 2, 4, 6],
        pairing_mode="target_driven",
        seed=13,
        drop_last=False,
    )

    batches = list(sampler)
    source_by_target = {
        int(target): int(source)
        for batch in batches
        for source, target in zip(batch.source_indices, batch.target_indices, strict=True)
    }

    assert len(sampler) == 2
    assert sampler.effective_pair_count == 4
    assert source_by_target == {1: 0, 2: 0, 4: 3, 6: 5}
    assert sorted(source_by_target) == [1, 2, 4, 6]


def test_target_driven_pairing_self_pairs_control_targets() -> None:
    sampler = _build_sampler(
        _build_metadata_index(),
        batch_size=3,
        config=_dataset_group_config("dataset", "celltype"),
        source_indices=[0, 3, 5],
        target_candidate_indices=[0, 1, 3, 4, 5, 6],
        pairing_mode="target_driven",
        seed=13,
        drop_last=False,
    )

    batches = list(sampler)
    pairs = {
        int(target): (int(source), int(label))
        for batch in batches
        for source, target, label in zip(
            batch.source_indices,
            batch.target_indices,
            batch.target_perturbation_ids,
            strict=True,
        )
    }

    assert sampler.effective_pair_count == 6
    assert pairs[0] == (0, 0)
    assert pairs[3] == (3, 0)
    assert pairs[5] == (5, 0)
    assert pairs[1][0] == 0
    assert pairs[4][0] == 3
    assert pairs[6][0] == 5


@pytest.mark.parametrize(
    ("source_indices", "target_candidate_indices", "message"),
    [
        ([0, 1], [2], "control-only source_positions"),
        ([3], [0, 4], "control targets to be present in source_positions"),
        ([0], [7], "no matched control source"),
    ],
)
def test_target_driven_pairing_rejects_invalid_pools(
    source_indices,
    target_candidate_indices,
    message,
) -> None:
    with pytest.raises(ValueError, match=message):
        _build_sampler(
            _build_metadata_index(),
            batch_size=2,
            config=_dataset_group_config("dataset", "celltype"),
            source_indices=source_indices,
            target_candidate_indices=target_candidate_indices,
            pairing_mode="target_driven",
            drop_last=False,
        )


def test_missing_target_pool_raises_explicit_error() -> None:
    sampler = _build_sampler(
        _build_metadata_index(),
        batch_size=4,
        config=_dataset_group_config("dataset", "celltype"),
        perturbed_target_policy="matched_control_cell",
    )

    with pytest.raises(RuntimeError, match="no matched control pool exists"):
        sampler.pair_source_positions([7])


def test_row_selection_intersects_all_requested_pools_and_preserves_order() -> None:
    selection = _prepare_perttf_metadata(
        _build_metadata_index(),
        config=_default_config(),
        row_indices=[3, 0, 1],
        source_indices=[1, 6, 0],
        target_candidate_indices=[0, 2, 3],
    ).row_selection

    np.testing.assert_array_equal(selection.base_indices, np.asarray([3, 0, 1], dtype=np.int64))
    np.testing.assert_array_equal(selection.source_positions, np.asarray([2, 1], dtype=np.int64))
    np.testing.assert_array_equal(selection.target_candidate_positions, np.asarray([1, 0], dtype=np.int64))


def test_prepared_metadata_compacts_rows_and_keeps_polars_frame() -> None:
    prepared = _prepare_perttf_metadata(
        _build_metadata_index(),
        config=_default_config(),
        row_indices=np.asarray([4, 0, 1], dtype=np.int64),
        source_indices=None,
        target_candidate_indices=None,
    )

    np.testing.assert_array_equal(prepared.row_selection.base_indices, np.asarray([4, 0, 1], dtype=np.int64))
    assert prepared.global_to_local == {4: 0, 0: 1, 1: 2}
    assert prepared.frame.columns[:4] == ["global_row_index", "dataset_index", "perturbation", "perturbation_id"]
    np.testing.assert_array_equal(
        _positions_for_global_rows(prepared, np.asarray([1, 4], dtype=np.int64)),
        np.asarray([2, 0], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        np.asarray(prepared.frame["celltype_id"])[
            _positions_for_global_rows(prepared, np.asarray([1, 4], dtype=np.int64))
        ],
        np.asarray([1, 0], dtype=np.int64),
    )
    assert tuple(prepared.frame["perturbation"].to_list()) == (
        "KO_C",
        "WT",
        "KO_A",
    )
    np.testing.assert_allclose(
        np.asarray(prepared.frame["size_factor"])[_positions_for_global_rows(prepared, np.asarray([1, 4], dtype=np.int64))],
        np.asarray([1.1, 1.0], dtype=np.float32),
    )


def test_prepared_metadata_uses_filtered_base_pool_for_compaction() -> None:
    sampler = _build_sampler(
        _build_metadata_index(),
        batch_size=1,
        config=_default_config(),
        row_indices=[3, 0, 1],
        source_indices=[1],
        target_candidate_indices=[0],
        drop_last=False,
    )

    np.testing.assert_array_equal(sampler.effective_source_positions, np.asarray([2], dtype=np.int64))
    np.testing.assert_array_equal(sampler.effective_target_candidate_positions, np.asarray([1], dtype=np.int64))


def test_source_only_subset_uses_filtered_base_pool_for_targets() -> None:
    sampler = _build_sampler(
        _build_two_row_metadata_index(),
        batch_size=1,
        config=_default_config(),
        source_indices=[0],
        seed=11,
        drop_last=False,
    )

    batch = sampler.pair_source_positions([0], seed=13)

    np.testing.assert_array_equal(sampler.effective_source_positions, np.asarray([0], dtype=np.int64))
    np.testing.assert_array_equal(
        sampler.effective_target_candidate_positions,
        np.asarray([0, 1], dtype=np.int64),
    )
    np.testing.assert_array_equal(batch.target_indices, np.asarray([1], dtype=np.int64))


def test_target_candidate_only_subset_is_allowed() -> None:
    sampler = _build_sampler(
        _build_two_row_metadata_index(),
        batch_size=2,
        config=_default_config(),
        target_candidate_indices=[1],
        drop_last=False,
    )

    batch = sampler.pair_source_positions([0, 1], seed=17)

    np.testing.assert_array_equal(sampler.effective_source_positions, np.asarray([0, 1], dtype=np.int64))
    np.testing.assert_array_equal(
        sampler.effective_target_candidate_positions,
        np.asarray([1], dtype=np.int64),
    )
    np.testing.assert_array_equal(batch.target_indices, np.asarray([1, 1], dtype=np.int64))


def test_default_non_perturbation_null_labels_are_encoded() -> None:
    sampler = _build_sampler(
        _build_null_context_metadata_index(),
        batch_size=1,
        config=_default_config(),
        drop_last=False,
    )

    batch = sampler.pair_source_positions([0], seed=19)

    np.testing.assert_array_equal(sampler.effective_source_positions, np.asarray([0, 1], dtype=np.int64))
    np.testing.assert_array_equal(batch.source_indices, np.asarray([0], dtype=np.int64))


def test_sampler_supports_perturbation_as_only_required_semantic_label() -> None:
    sampler = _build_sampler(
        _build_metadata_index(),
        batch_size=1,
        config=_default_config(label_fields={"perturb_label": "perturbation"}),
        source_indices=[0],
        target_candidate_indices=[6],
        seed=23,
        drop_last=False,
    )

    batch = sampler.pair_source_positions([0], seed=29)

    np.testing.assert_array_equal(batch.target_indices, np.asarray([6], dtype=np.int64))
    np.testing.assert_array_equal(batch.target_perturbation_ids, np.asarray([1], dtype=np.int64))


def test_custom_drop_null_labels_removes_selected_base_pool() -> None:
    with pytest.raises(ValueError, match="metadata row pool is empty"):
        _build_sampler(
            _build_null_context_metadata_index(),
            batch_size=1,
            config=_default_config(drop_null_labels=("celltype",)),
            drop_last=False,
        )


def test_pair_source_positions_rejects_rows_outside_configured_source_pool() -> None:
    sampler = _build_sampler(
        _build_metadata_index(),
        batch_size=2,
        config=_default_config(),
        source_indices=[0, 1, 5, 6],
    )

    with pytest.raises(
        ValueError,
        match="configured source pool",
    ):
        sampler.pair_source_positions([2])


def test_asymmetric_target_candidate_pool_preserves_pairing_invariants() -> None:
    sampler = _build_sampler(
        _build_metadata_index(),
        batch_size=3,
        config=_dataset_group_config("dataset", "celltype"),
        source_indices=[0, 3, 6],
        target_candidate_indices=[2, 4, 5],
        perturbed_target_policy="matched_control_cell",
        seed=13,
    )

    batch = sampler.pair_source_positions([0, 3, 6], seed=19)

    np.testing.assert_array_equal(batch.target_indices, np.asarray([2, 4, 5], dtype=np.int64))
    assert set(batch.target_indices.tolist()).issubset({2, 4, 5})


def test_restricted_target_pool_can_make_source_unpairable() -> None:
    sampler = _build_sampler(
        _build_metadata_index(),
        batch_size=1,
        config=_default_config(),
        source_indices=[6],
        target_candidate_indices=[5],
    )

    with pytest.raises(
        RuntimeError,
        match="configured target pool",
    ):
        sampler.pair_source_positions([6])


def test_sampler_iteration_is_seed_deterministic_and_preserves_pairing_invariants() -> None:
    config = _dataset_group_config("dataset", "celltype")
    sampler_a = _build_sampler(
        _build_metadata_index(),
        batch_size=3,
        config=config,
        seed=29,
        drop_last=False,
        perturbed_target_policy="matched_control_cell",
        source_indices=[0, 1, 2, 3, 4, 5, 6],
    )
    sampler_b = _build_sampler(
        _build_metadata_index(),
        batch_size=3,
        config=config,
        seed=29,
        drop_last=False,
        perturbed_target_policy="matched_control_cell",
        source_indices=[0, 1, 2, 3, 4, 5, 6],
    )

    batches_a = list(sampler_a)
    batches_b = list(sampler_b)

    assert len(batches_a) == len(batches_b)
    for batch_a, batch_b in zip(batches_a, batches_b, strict=True):
        np.testing.assert_array_equal(batch_a.source_indices, batch_b.source_indices)
        np.testing.assert_array_equal(batch_a.target_indices, batch_b.target_indices)
        np.testing.assert_array_equal(batch_a.target_perturbation_ids, batch_b.target_perturbation_ids)


def test_sampler_set_epoch_is_repeatable_and_changes_pair_sequence() -> None:
    sampler = _build_sampler(
        _build_metadata_index(),
        batch_size=1,
        config=_default_config(),
        source_indices=[0, 1, 5, 6],
        seed=17,
        drop_last=True,
    )

    sampler.set_epoch(0)
    epoch0_first = [
        (tuple(batch.source_indices.tolist()), tuple(batch.target_indices.tolist()))
        for batch in sampler
    ]
    sampler.set_epoch(0)
    epoch0_second = [
        (tuple(batch.source_indices.tolist()), tuple(batch.target_indices.tolist()))
        for batch in sampler
    ]
    sampler.set_epoch(1)
    epoch1 = [
        (tuple(batch.source_indices.tolist()), tuple(batch.target_indices.tolist()))
        for batch in sampler
    ]

    assert epoch0_first == epoch0_second
    assert epoch1 != epoch0_first
