from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
from scipy import sparse
from scipy.stats import t as student_t


def normalize_expression(values, target_sum=10000.0, input_scale="counts"):
    """Normalize count or log1p expression to one fixed library size."""
    if sparse.issparse(values):
        values = values.toarray()
    values = np.asarray(values, dtype=np.float64)
    if input_scale == "log1p":
        values = np.expm1(values)
    elif input_scale != "counts":
        raise ValueError("input_scale must be 'counts' or 'log1p'")

    values = np.clip(values, 0, None)
    totals = values.sum(axis=1, keepdims=True)
    normalized = np.zeros_like(values)
    np.divide(values, totals, out=normalized, where=totals > 0)
    return np.log1p(normalized * target_sum)


def group_moments_from_anndata(
    adata,
    indices,
    context_col="celltype",
    perturbation_col="genotype",
    layer=None,
    input_scale="log1p",
    target_sum=10000.0,
    chunk_size=1024,
):
    """Build normalized group moments from selected AnnData rows."""
    indices = np.asarray(indices, dtype=np.int64)
    moments = GroupMoments()
    matrix = adata.X if layer is None else adata.layers[layer]
    for start in range(0, indices.size, chunk_size):
        chunk = indices[start : start + chunk_size]
        values = normalize_expression(
            matrix[chunk], target_sum=target_sum, input_scale=input_scale
        )
        moments.update(
            values,
            adata.obs.iloc[chunk][context_col].to_numpy(),
            adata.obs.iloc[chunk][perturbation_col].to_numpy(),
        )
    return moments


@dataclass
class GroupStats:
    n: int
    mean: np.ndarray
    variance: np.ndarray


class GroupMoments:
    """Streaming gene-wise moments grouped by (context, perturbation)."""

    def __init__(self):
        self._moments = {}

    def update(self, values, contexts: Iterable, perturbations: Iterable):
        values = np.asarray(values, dtype=np.float64)
        contexts = np.asarray(contexts)
        perturbations = np.asarray(perturbations)
        if values.ndim != 2 or values.shape[0] != contexts.size or contexts.size != perturbations.size:
            raise ValueError("Expression rows, contexts, and perturbations must align")

        keys = np.asarray(
            [(str(context), str(perturbation)) for context, perturbation in zip(contexts, perturbations)],
            dtype=object,
        )
        for key in sorted(set(map(tuple, keys)), key=str):
            mask = np.fromiter((tuple(row) == key for row in keys), dtype=bool)
            block = values[mask]
            n, value_sum, square_sum = self._moments.get(
                key,
                (0, np.zeros(values.shape[1], dtype=np.float64), np.zeros(values.shape[1], dtype=np.float64)),
            )
            self._moments[key] = (
                n + block.shape[0],
                value_sum + block.sum(axis=0),
                square_sum + np.square(block).sum(axis=0),
            )

    def finalize(self) -> Dict[Tuple[str, str], GroupStats]:
        groups = {}
        for key, (n, value_sum, square_sum) in self._moments.items():
            mean = value_sum / n
            variance = (
                np.maximum((square_sum - np.square(value_sum) / n) / (n - 1), 0)
                if n > 1
                else np.zeros_like(mean)
            )
            groups[key] = GroupStats(n=n, mean=mean, variance=variance)
        return groups


def benjamini_hochberg(pvalues):
    pvalues = np.asarray(pvalues, dtype=np.float64)
    order = np.argsort(pvalues)
    ranked = pvalues[order]
    adjusted = np.minimum.accumulate((ranked * pvalues.size / np.arange(1, pvalues.size + 1))[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0, 1)
    return result


def welch_pvalues(group_a: GroupStats, group_b: GroupStats):
    difference = group_a.mean - group_b.mean
    var_a = group_a.variance / group_a.n
    var_b = group_b.variance / group_b.n
    standard_error_sq = var_a + var_b
    denominator = np.square(var_a) / max(group_a.n - 1, 1) + np.square(var_b) / max(group_b.n - 1, 1)

    pvalues = np.ones_like(difference)
    valid = (standard_error_sq > 0) & (denominator > 0)
    statistic = np.zeros_like(difference)
    statistic[valid] = difference[valid] / np.sqrt(standard_error_sq[valid])
    degrees_freedom = np.zeros_like(difference)
    degrees_freedom[valid] = np.square(standard_error_sq[valid]) / denominator[valid]
    pvalues[valid] = 2 * student_t.sf(np.abs(statistic[valid]), degrees_freedom[valid])
    pvalues[(standard_error_sq == 0) & (difference != 0)] = 0.0
    return pvalues


def compute_perturbation_metrics(
    real_groups: Dict[Tuple[str, str], GroupStats],
    predicted_groups: Dict[Tuple[str, str], GroupStats],
    control_value="WT",
    fdr_threshold=0.05,
    min_cells=3,
):
    """Compute perturbation metrics per context and macro-average them."""
    per_group = {}
    for (context, perturbation), predicted in predicted_groups.items():
        if perturbation == control_value:
            continue
        real = real_groups.get((context, perturbation))
        control = real_groups.get((context, control_value))
        if real is None or control is None or min(real.n, predicted.n, control.n) < min_cells:
            continue

        real_delta = real.mean - control.mean
        predicted_delta = predicted.mean - control.mean
        if np.std(real_delta) == 0 or np.std(predicted_delta) == 0:
            pearson_delta = float("nan")
        else:
            pearson_delta = float(np.corrcoef(real_delta, predicted_delta)[0, 1])

        real_fdr = benjamini_hochberg(welch_pvalues(real, control))
        predicted_fdr = benjamini_hochberg(welch_pvalues(predicted, control))
        real_significant = np.flatnonzero(real_fdr < fdr_threshold)
        predicted_significant = np.flatnonzero(predicted_fdr < fdr_threshold)
        n_real_degs = real_significant.size

        if n_real_degs:
            real_ranked = real_significant[np.argsort(np.abs(real_delta[real_significant]))[::-1]]
            predicted_ranked = predicted_significant[
                np.argsort(np.abs(predicted_delta[predicted_significant]))[::-1]
            ][:n_real_degs]
            overlap = np.intersect1d(real_ranked, predicted_ranked).size / n_real_degs
            direction_match = np.mean(
                np.sign(real_delta[real_significant]) == np.sign(predicted_delta[real_significant])
            )
        else:
            overlap = 0.0
            direction_match = float("nan")

        per_group[f"{context}|{perturbation}"] = {
            "pearson_delta": pearson_delta,
            "mse_delta": float(np.mean(np.square(real_delta - predicted_delta))),
            "ttest_de_overlap_at_n": float(overlap),
            "ttest_de_direction_match": float(direction_match),
            "n_real_degs": int(n_real_degs),
            "n_real_cells": int(real.n),
            "n_predicted_cells": int(predicted.n),
        }

    metric_names = [
        "pearson_delta",
        "mse_delta",
        "ttest_de_overlap_at_n",
        "ttest_de_direction_match",
        "n_real_degs",
    ]
    aggregate = {}
    for name in metric_names:
        values = np.asarray([result[name] for result in per_group.values()], dtype=np.float64)
        aggregate[name] = float(np.nanmean(values)) if values.size and not np.isnan(values).all() else float("nan")
    aggregate["n_evaluated_groups"] = len(per_group)
    return {"per_group": per_group, "aggregate": aggregate}


def compute_metrics_from_anndata(
    real_adata,
    predicted_adata,
    prediction_key="mvc_next_expr",
    context_col="celltype",
    real_perturbation_col="genotype",
    predicted_perturbation_col="genotype_next",
    real_layer=None,
    real_scale="log1p",
    prediction_scale="counts",
    target_sum=10000.0,
    control_value="WT",
    fdr_threshold=0.05,
    min_cells=3,
):
    """Apply the same metric core to real and predicted AnnData outputs."""
    real_groups = group_moments_from_anndata(
        real_adata,
        np.arange(real_adata.n_obs),
        context_col=context_col,
        perturbation_col=real_perturbation_col,
        layer=real_layer,
        input_scale=real_scale,
        target_sum=target_sum,
    ).finalize()
    predicted_values = predicted_adata.obsm[prediction_key]
    predicted_values = normalize_expression(
        predicted_values,
        target_sum=target_sum,
        input_scale=prediction_scale,
    )
    predicted_moments = GroupMoments()
    predicted_moments.update(
        predicted_values,
        predicted_adata.obs[context_col].to_numpy(),
        predicted_adata.obs[predicted_perturbation_col].to_numpy(),
    )
    return compute_perturbation_metrics(
        real_groups,
        predicted_moments.finalize(),
        control_value=control_value,
        fdr_threshold=fdr_threshold,
        min_cells=min_cells,
    )


def prediction_scale(distribution):
    return "counts" if distribution not in {None, "zig"} else "log1p"


def labels_to_names(labels, index_to_name):
    if hasattr(labels, "detach"):
        labels = labels.detach().cpu().numpy()
    return np.asarray([index_to_name.get(int(value), str(int(value))) for value in labels])
