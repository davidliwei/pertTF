"""Helpers for parsing and validating delimited multi-perturbation labels.

A cell may carry several perturbations written as one string such as ``"A+B"``.
With ``delimiter=None`` every helper treats labels as opaque legacy single labels.
"""
import warnings
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

PAD = -1


def _is_missing(label) -> bool:
    """True for None, pd.NA or a float NaN (pandas missing values)."""
    return label is None or label is pd.NA or (isinstance(label, float) and np.isnan(label))


def _raise_if_missing(labels, where: str) -> None:
    n_missing = sum(1 for label in labels if _is_missing(label))
    if n_missing:
        raise ValueError(
            f"{where} has {n_missing} missing (NaN/None) perturbation labels; "
            "multi-perturbation mode needs a label for every cell (use the control label for controls)")


def split_perturbation_label(label: str, delimiter: Optional[str], control: str = "WT") -> tuple:
    """Split a label into unique components (first-occurrence order), dropping
    ``control`` unless it is the only component."""
    if delimiter is None:
        return (label,)
    parts = label.split(delimiter)
    for p in parts:
        if p == "" or p.strip() != p:
            raise ValueError(f"Malformed perturbation label {label!r}: empty or whitespace-padded component {p!r}")
    unique = list(dict.fromkeys(parts))
    non_control = [p for p in unique if p != control]
    return tuple(non_control) if non_control else (control,)


def canonical_label(label: str, delimiter: Optional[str], control: str = "WT") -> str:
    """Sorted components joined by the delimiter; identity when ``delimiter`` is None."""
    if delimiter is None:
        return label
    return delimiter.join(sorted(split_perturbation_label(label, delimiter, control)))


def unique_perturbations(labels: Iterable[str], delimiter: Optional[str], control: str = "WT") -> list:
    """Single perturbations in first-occurrence order across all labels."""
    return list(dict.fromkeys(
        c for label in labels for c in split_perturbation_label(label, delimiter, control)
    ))


def labels_in_mapping(labels: Iterable[str], mapping: Mapping[str, int], delimiter: Optional[str],
                      control: str = "WT") -> np.ndarray:
    """Boolean per label, True only when every component is a key of ``mapping``."""
    return np.array(
        [all(c in mapping for c in split_perturbation_label(label, delimiter, control)) for label in labels],
        dtype=bool,
    )


def labels_to_padded_array(label: str, mapping: Mapping[str, int], delimiter: Optional[str], width: int,
                           control: str = "WT", drop_unknown: bool = False) -> np.ndarray:
    """int64 array of shape ``(width,)``: component indices followed by -1 padding.

    With ``drop_unknown`` components missing from ``mapping`` become padding instead of raising
    (an all-unknown label gives an all-padding row)."""
    parts = split_perturbation_label(label, delimiter, control)
    if len(parts) > width:
        raise ValueError(f"Label {label!r} has {len(parts)} perturbations, more than width {width}")
    unknown = [p for p in parts if p not in mapping]
    if unknown:
        if not drop_unknown:
            raise ValueError(f"Label {label!r} has components not in mapping: {unknown}")
        parts = tuple(p for p in parts if p in mapping)
    out = np.full(width, PAD, dtype=np.int64)
    out[: len(parts)] = [mapping[p] for p in parts]
    return out


def max_set_size(labels: Iterable[str], delimiter: Optional[str], control: str = "WT") -> int:
    """Largest component count over ``labels`` (at least 1)."""
    return max([1] + [len(split_perturbation_label(label, delimiter, control)) for label in labels])


def validate_perturbation_labels(labels: Iterable[str], delimiter: Optional[str], control: str = "WT",
                                 existing_mapping: Optional[Mapping[str, int]] = None,
                                 max_perts_per_cell: Optional[int] = None) -> None:
    """Check labels against the delimiter configuration; raises ValueError on problems."""
    labels = list(labels)
    if delimiter is None:
        if any(isinstance(label, str) and "+" in label for label in labels):
            warnings.warn(
                "Some perturbation labels contain '+' but no multi-perturbation delimiter is set; "
                "they are treated as single legacy labels.", UserWarning, stacklevel=2)
        return
    _raise_if_missing(labels, "Perturbation label column")
    malformed = []
    for label in dict.fromkeys(labels):
        try:
            split_perturbation_label(label, delimiter, control)
        except ValueError:
            malformed.append(label)
    if malformed:
        raise ValueError(f"Malformed perturbation labels: {malformed}")
    if existing_mapping is not None:
        bad_keys = [k for k in existing_mapping if delimiter in k]
        if bad_keys:
            raise ValueError(f"Existing mapping keys contain the delimiter {delimiter!r}: {bad_keys}")
    if max_perts_per_cell is not None:
        too_big = [label for label in dict.fromkeys(labels)
                   if len(split_perturbation_label(label, delimiter, control)) > max_perts_per_cell]
        if too_big:
            raise ValueError(f"Labels exceed max_perts_per_cell={max_perts_per_cell}: {too_big}")


def canonicalize_perturbation_columns(adata, delimiter: Optional[str], control: str = "WT",
                                      columns: Sequence[str] = ("genotype", "genotype_next")) -> int:
    """Rewrite each present obs column in place with canonical labels (object dtype).

    Returns the number of changed cells; does nothing when ``delimiter`` is None.
    """
    if delimiter is None:
        return 0
    changed = 0
    for col in columns:
        if col not in adata.obs.columns:
            continue
        _raise_if_missing(adata.obs[col].tolist(), f"obs column {col!r}")
        old = adata.obs[col].astype(str).to_numpy(dtype=object)
        cache = {}
        new = np.empty(len(old), dtype=object)
        for i, label in enumerate(old):
            if label not in cache:
                cache[label] = canonical_label(label, delimiter, control)
            new[i] = cache[label]
        changed += int((old != new).sum())
        adata.obs[col] = new
    if changed > 0:
        print(f"Canonicalized {changed} perturbation labels in columns {list(columns)}")
    return changed
