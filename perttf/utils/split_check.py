import warnings

import numpy as np
import pandas as pd

from perttf.utils.pert_labels import split_perturbation_label


def _check_indices(name, indices, n_obs):
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional collection")
    if indices.min() < 0 or indices.max() >= n_obs:
        raise IndexError(f"{name} contain indices outside the AnnData bounds")
    if np.unique(indices).size != indices.size:
        raise ValueError(f"{name} contain duplicate indices")
    return indices


def _warn_groups(message, groups, limit=10):
    preview = ", ".join(str(group) for group in groups[:limit])
    suffix = f", ... {len(groups) - limit} more" if len(groups) > limit else ""
    warnings.warn(f"{message}: {preview}{suffix}", stacklevel=2)


def check_train_valid_split(
    adata,
    train_indices,
    valid_indices,
    mode,
    control_value="WT",
    check_columns=None,
    min_cells=30,
    context_col="celltype",
    perturbation_col="genotype",
    perturbation_unit="set",
    pert_delimiter=None,
):
    """Check train/validation index coverage and summarize perturbation holdouts."""
    if mode not in {"identity", "lochness", "pert"}:
        raise ValueError("mode must be 'identity', 'lochness', or 'pert'")
    if perturbation_unit not in {"set", "single"}:
        raise ValueError("perturbation_unit must be 'set' or 'single'")

    train_indices = _check_indices("train_indices", train_indices, adata.n_obs)
    valid_indices = _check_indices("valid_indices", valid_indices, adata.n_obs)
    overlap = np.intersect1d(train_indices, valid_indices)
    report = {
        "mode": mode,
        "n_train": int(train_indices.size),
        "n_valid": int(valid_indices.size),
    }

    if mode != "pert":
        if overlap.size:
            raise ValueError(f"Training and validation contain {overlap.size} overlapping rows")
        if check_columns is None:
            columns = [context_col, perturbation_col]
        else:
            columns = [check_columns] if isinstance(check_columns, str) else list(check_columns)
        missing_columns = [column for column in columns if column not in adata.obs]
        if missing_columns:
            raise ValueError(f"AnnData is missing split-check columns: {missing_columns}")

        # With multi-perturbation labels in single-unit mode the model sees components, and whole
        # set labels are ~unique per cell at high MOI: compare components, and only warn about
        # validation-only ones (the pert-mode check reports them as unseen_perturbation).
        by_component = pert_delimiter is not None and perturbation_unit == "single"

        def _values(obs, column):
            values = set(obs[column].dropna().unique())
            if by_component and column == perturbation_col:
                values = {
                    component
                    for label in values
                    for component in split_perturbation_label(str(label), pert_delimiter, control_value)
                }
            return values

        column_coverage = {}
        for column in columns:
            if adata.obs.iloc[np.concatenate([train_indices, valid_indices])][column].isna().any():
                raise ValueError(f"Split-check column {column!r} contains missing values")
            train_values = _values(adata.obs.iloc[train_indices], column)
            valid_values = _values(adata.obs.iloc[valid_indices], column)
            valid_only = sorted(valid_values - train_values, key=str)
            train_only = sorted(train_values - valid_values, key=str)
            if valid_only:
                if by_component and column == perturbation_col:
                    _warn_groups(f"Validation {column!r} components absent from training", valid_only)
                else:
                    raise ValueError(f"Validation {column!r} values are absent from training: {valid_only}")
            if train_only:
                _warn_groups(f"Training {column!r} values absent from validation", train_only)
            column_coverage[column] = {
                "valid_only": valid_only,
                "train_only": train_only,
            }
        report["column_coverage"] = column_coverage
        return report

    required_columns = [context_col, perturbation_col]
    missing_columns = [column for column in required_columns if column not in adata.obs]
    if missing_columns:
        raise ValueError(f"AnnData is missing perturbation split columns: {missing_columns}")
    selected_obs = adata.obs.iloc[np.unique(np.concatenate([train_indices, valid_indices]))]
    if selected_obs[required_columns].isna().any().any():
        raise ValueError("Perturbation split context or perturbation labels contain missing values")
    if control_value not in set(selected_obs[perturbation_col]):
        raise ValueError(f"Control value {control_value!r} is absent from the train/validation rows")

    train_obs = adata.obs.iloc[train_indices]
    valid_obs = adata.obs.iloc[valid_indices]
    train_is_control = train_obs[perturbation_col].to_numpy() == control_value
    valid_is_control = valid_obs[perturbation_col].to_numpy() == control_value
    train_controls = train_indices[train_is_control]
    train_targets = train_indices[~train_is_control]
    valid_controls = valid_indices[valid_is_control]
    valid_targets = valid_indices[~valid_is_control]
    if not train_controls.size or not train_targets.size:
        raise ValueError("Perturbation training requires both control and perturbed rows")
    if not valid_controls.size or not valid_targets.size:
        raise ValueError("Perturbation validation requires both control and perturbed rows")

    overlapping_controls = np.intersect1d(train_controls, valid_controls)
    overlapping_targets = np.intersect1d(train_targets, valid_targets)
    if overlapping_targets.size:
        raise ValueError(
            f"Training and validation contain {overlapping_targets.size} overlapping perturbed target rows"
        )

    train_contexts = set(train_obs[context_col].unique())
    train_target_obs = train_obs.loc[~train_is_control]
    valid_control_obs = valid_obs.loc[valid_is_control]
    valid_target_obs = valid_obs.loc[~valid_is_control]
    valid_control_contexts = set(valid_control_obs[context_col].unique())
    valid_target_contexts = set(valid_target_obs[context_col].unique())
    incomplete_contexts = sorted(valid_control_contexts ^ valid_target_contexts, key=str)
    if incomplete_contexts:
        raise ValueError(
            "Every validation context must contain controls and perturbed targets; "
            f"incomplete contexts: {incomplete_contexts}"
        )

    expand = pert_delimiter is not None and perturbation_unit == "single"

    def _target_pairs(obs):
        """(context, perturbation) per target row; one row per component in single mode."""
        contexts = obs[context_col].to_numpy()
        labels = obs[perturbation_col].to_numpy()
        if not expand:
            return list(zip(contexts, labels))
        cache = {}
        pairs = []
        for context, label in zip(contexts, labels):
            if label not in cache:
                cache[label] = split_perturbation_label(str(label), pert_delimiter, control_value)
            pairs.extend((context, component) for component in cache[label])
        return pairs

    train_pairs = _target_pairs(train_target_obs)
    valid_pairs = _target_pairs(valid_target_obs)
    train_perturbations = {pert for _, pert in train_pairs}
    train_combinations = set(train_pairs)
    valid_combinations = set(valid_pairs)
    combination_types = {
        "seen_combination": [],
        "context_transfer": [],
        "unseen_perturbation": [],
        "unseen_context": [],
        "unseen_context_and_perturbation": [],
    }
    track_unseen_combination = pert_delimiter is not None and perturbation_unit == "set"
    if track_unseen_combination:
        combination_types["unseen_combination"] = []
        train_components = {
            component
            for label in train_perturbations
            for component in split_perturbation_label(str(label), pert_delimiter, control_value)
        }
    for context, perturbation in sorted(valid_combinations, key=lambda pair: (str(pair[0]), str(pair[1]))):
        if (context, perturbation) in train_combinations:
            category = "seen_combination"
        elif context in train_contexts and perturbation in train_perturbations:
            category = "context_transfer"
        elif (
            track_unseen_combination
            and context in train_contexts
            and all(
                c in train_components
                for c in split_perturbation_label(str(perturbation), pert_delimiter, control_value)
            )
        ):
            category = "unseen_combination"
        elif context in train_contexts:
            category = "unseen_perturbation"
        elif perturbation in train_perturbations:
            category = "unseen_context"
        else:
            category = "unseen_context_and_perturbation"
        combination_types[category].append((context, perturbation))

    train_control_contexts = set(train_obs.loc[train_is_control, context_col].unique())
    train_target_contexts = set(train_target_obs[context_col].unique())
    unpaired_train_contexts = sorted(train_control_contexts ^ train_target_contexts, key=str)
    if unpaired_train_contexts:
        _warn_groups("Training contexts without both control and perturbed rows", unpaired_train_contexts)

    context_exposure = {
        "seen_combination": [],
        "fewshot": [],
        "control_only": [],
        "unseen": [],
    }
    for context in sorted(valid_target_contexts, key=str):
        context_combinations = {
            pair for pair in valid_combinations if pair[0] == context
        }
        if context not in train_contexts:
            category = "unseen"
        elif context not in train_target_contexts:
            category = "control_only"
        elif context_combinations.issubset(train_combinations):
            category = "seen_combination"
        else:
            category = "fewshot"
        context_exposure[category].append(context)

    min_cells = int(min_cells)
    low_control_groups = [
        (context, int(count))
        for context, count in valid_control_obs.groupby(context_col, observed=True).size().items()
        if count < min_cells
    ]
    if expand:
        valid_group_counts = pd.Series(valid_pairs).value_counts()
        group_items = sorted(valid_group_counts.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1])))
    else:
        group_items = list(valid_target_obs.groupby([context_col, perturbation_col], observed=True).size().items())
    low_target_groups = [
        (context, perturbation, int(count))
        for (context, perturbation), count in group_items
        if count < min_cells
    ]
    if track_unseen_combination:
        set_counts = valid_obs.groupby([context_col, perturbation_col], observed=True).size()
        set_counts = set_counts[set_counts > 0]
        if len(set_counts) and int((set_counts >= min_cells).sum()) * 2 < len(set_counts):
            warnings.warn(
                f"Fewer than half of validation (context, perturbation set) groups have at least "
                f"{min_cells} cells; consider perturbation_unit='single' to evaluate per component "
                "perturbation.",
                stacklevel=2,
            )
    if low_control_groups:
        _warn_groups(f"Validation control groups with fewer than {min_cells} cells", low_control_groups)
    if low_target_groups:
        _warn_groups(
            f"Validation perturbation-context groups with fewer than {min_cells} cells",
            low_target_groups,
        )

    report.update({
        "n_valid_controls": int(valid_controls.size),
        "n_valid_targets": int(valid_targets.size),
        "n_overlapping_controls": int(overlapping_controls.size),
        "combination_types": combination_types,
        "combination_type_counts": {
            name: len(combinations) for name, combinations in combination_types.items()
        },
        "context_exposure": context_exposure,
        "overlapping_control_contexts": sorted(
            set(adata.obs.iloc[overlapping_controls][context_col].unique()), key=str
        ),
        "incomplete_validation_contexts": incomplete_contexts,
        "unpaired_train_contexts": unpaired_train_contexts,
        "low_control_groups": low_control_groups,
        "low_target_groups": low_target_groups,
    })
    return report
