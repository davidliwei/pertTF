from dataclasses import dataclass
import os
import pickle
import random
from typing import Literal, Optional

import numpy as np


@dataclass(frozen=True)
class PerturbationPair:
    source_idx: int
    target_idx: int
    condition_label: str
    transition_type: Literal["identity", "perturbation"]


class ContextAwareOT:
    """Compute and sample directional OT pair maps for eligible cells."""

    def __init__(self, adata, source_indices, target_indices, ot_params):
        self.adata = adata
        self.source_indices = source_indices
        self.target_indices = target_indices
        self.ot_params = ot_params
        self.forward = {}
        self.reverse = {}
        self._compute()

    @staticmethod
    def _weighted_choice(values, weights, rng=None):
        weights = np.asarray(weights, dtype=float)
        if len(values) == 0 or weights.sum() <= 0:
            raise ValueError("Cannot sample from an empty OT pairing pool")
        if isinstance(rng, np.random.Generator):
            return rng.choice(values, p=weights / weights.sum())
        return random.choices(values, weights=weights.tolist(), k=1)[0]

    def _compute(self):
        try:
            from .ot import compute_ot_for_subset
        except ImportError as error:
            raise ImportError("OT pairing requires ott-jax; install it or set use_ot=False") from error

        pairing_indices = np.unique(np.concatenate([self.source_indices, self.target_indices]))
        self.forward = compute_ot_for_subset(
            self.adata[pairing_indices],
            top_k=self.ot_params.get("ot_top_k", 10),
            epsilon=self.ot_params.get("ot_epsilon", "auto"),
            max_dist_sq=self.ot_params.get("ot_max_dist", "auto"),
            red_key="X_pca",
            epsilon_scaler=self.ot_params.get("ot_epsilon_scaler", 0.01),
        )

        allowed_sources = set(self.adata.obs.index[self.source_indices])
        allowed_targets = set(self.adata.obs.index[self.target_indices])
        for source_id, perturbations in self.forward.items():
            if source_id not in allowed_sources:
                continue
            for target_ids, weights in perturbations.values():
                for target_id, weight in zip(target_ids, weights):
                    if target_id in allowed_targets:
                        source_ids, source_weights = self.reverse.setdefault(target_id, ([], []))
                        source_ids.append(source_id)
                        source_weights.append(float(weight))

        cache_path = self.ot_params.get("ot_pickle_path")
        if cache_path:
            full_cache = {}
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, "rb") as file:
                        full_cache = pickle.load(file)
                except Exception:
                    pass
            full_cache.update(self.forward)
            try:
                with open(cache_path, "wb") as file:
                    pickle.dump(full_cache, file)
            except Exception as error:
                print(f"Warning: Could not save OT pickle: {error}")

    def sample_target(self, source_idx, perturbation, rng=None):
        pair = self.forward.get(self.adata.obs.index[source_idx], {}).get(perturbation)
        if pair is None:
            return None
        target_ids, weights = pair
        allowed_targets = set(self.adata.obs.index[self.target_indices])
        keep = np.asarray([target_id in allowed_targets for target_id in target_ids])
        if not keep.any():
            return None
        target_id = self._weighted_choice(np.asarray(target_ids)[keep], np.asarray(weights)[keep], rng)
        return int(self.adata.obs.index.get_loc(target_id))

    def sample_source(self, target_idx, rng=None):
        pair = self.reverse.get(self.adata.obs.index[target_idx])
        if pair is None:
            raise ValueError(f"No OT source pairing available for target {self.adata.obs.index[target_idx]!r}")
        source_ids, weights = pair
        source_id = self._weighted_choice(source_ids, weights, rng)
        return int(self.adata.obs.index.get_loc(source_id))


class ContextAwarePairing:
    """Resolve source-target pairs under one context-aware pairing policy."""

    def __init__(
        self,
        adata,
        indices,
        source_indices=None,
        target_indices=None,
        pairing_anchor: Literal["source", "target"] = "source",
        source_selection: Literal["strict", "all"] = "all",
        target_selection: Literal["strict", "all"] = "all",
        identity_condition: Optional[Literal["source"]] = "source",
        pair_schedule: Literal["dynamic", "fixed"] = "dynamic",
        target_sampling: Literal["cell", "perturbation"] = "perturbation",
        seed=None,
        control_value="WT",
        perturbation_mode=True,
        ot_params=None,
    ):
        self.adata = adata
        self.pairing_anchor = pairing_anchor
        self.source_selection = source_selection
        self.target_selection = target_selection
        self.identity_condition = identity_condition
        self.pair_schedule = pair_schedule
        self.target_sampling = target_sampling
        self.seed = seed
        self.control_value = control_value
        self.perturbation_mode = perturbation_mode
        self._configure_indices(indices, source_indices, target_indices)

        if ot_params is not None and pairing_anchor == "target" and source_selection != "strict":
            raise ValueError("Target-driven OT pairing requires source_selection='strict' because OT sources are controls")
        self.ot = ContextAwareOT(adata, self.source_indices, self.target_indices, ot_params) if ot_params is not None else None
        self.target_cell_pool = self._build_target_cell_pool() if pairing_anchor == "source" and perturbation_mode else None
        self.source_cell_pool = self._build_source_cell_pool() if pairing_anchor == "target" and perturbation_mode and self.ot is None else None
        self._fixed_pairs = self._build_fixed_pairs() if pair_schedule == "fixed" else None

    def _configure_indices(self, indices, source_indices, target_indices):
        allowed = {
            "pairing_anchor": (self.pairing_anchor, {"source", "target"}),
            "source_selection": (self.source_selection, {"strict", "all"}),
            "target_selection": (self.target_selection, {"strict", "all"}),
            "pair_schedule": (self.pair_schedule, {"dynamic", "fixed"}),
            "target_sampling": (self.target_sampling, {"cell", "perturbation"}),
        }
        for name, (value, choices) in allowed.items():
            if value not in choices:
                raise ValueError(f"{name} must be one of {sorted(choices)}")
        if self.identity_condition not in {"source", None}:
            raise ValueError("identity_condition must be 'source' or None")
        if self.pairing_anchor == "target" and not self.perturbation_mode:
            raise ValueError("Target-driven pairing requires perturbation prediction")

        indices = np.asarray(indices, dtype=np.int64)
        source_indices = np.asarray(indices if source_indices is None else source_indices, dtype=np.int64)
        target_indices = np.asarray(indices if target_indices is None else target_indices, dtype=np.int64)
        for name, values in (("source_indices", source_indices), ("target_indices", target_indices)):
            if values.size and (values.min() < 0 or values.max() >= self.adata.n_obs):
                raise IndexError(f"{name} contain indices outside the AnnData bounds")

        source_perts = self.adata.obs.iloc[source_indices]["genotype"].to_numpy()
        target_perts = self.adata.obs.iloc[target_indices]["genotype"].to_numpy()
        self.source_indices = source_indices[source_perts == self.control_value] if self.source_selection == "strict" else source_indices
        self.target_indices = target_indices[target_perts != self.control_value] if self.target_selection == "strict" else target_indices
        self.anchor_indices = self.source_indices if self.pairing_anchor == "source" else self.target_indices
        if self.pair_schedule == "fixed" and self.anchor_indices.size == 0:
            raise ValueError("Fixed pairing requires at least one eligible anchor cell")

    @staticmethod
    def _choice(values, rng=None):
        if len(values) == 0:
            raise ValueError("Cannot sample from an empty pairing pool")
        if isinstance(rng, np.random.Generator):
            return values[int(rng.integers(len(values)))]
        return random.choice(values)

    def _build_target_cell_pool(self):
        pool = {}
        for target_idx in self.target_indices:
            target = self.adata.obs.iloc[target_idx]
            pool.setdefault(target["celltype"], {}).setdefault(target["genotype"], []).append(int(target_idx))
        return pool

    def _build_source_cell_pool(self):
        pool = {}
        for source_idx in self.source_indices:
            source = self.adata.obs.iloc[source_idx]
            pool.setdefault(source["celltype"], []).append(int(source_idx))
        return pool

    def _identity_pair(self, source_idx):
        source_label = self.adata.obs.iloc[source_idx]["genotype"]
        condition_label = self.control_value if self.identity_condition is None else source_label
        return PerturbationPair(int(source_idx), int(source_idx), condition_label, "identity")

    def _resolve_source_pair(self, source_idx, rng=None):
        source = self.adata.obs.iloc[source_idx]
        if not self.perturbation_mode or source["genotype"] != self.control_value:
            return self._identity_pair(source_idx)

        target_groups = (self.target_cell_pool or {}).get(source["celltype"], {})
        if not target_groups:
            return self._identity_pair(source_idx)
        if self.target_sampling == "cell":
            target_idx = int(self._choice([idx for group in target_groups.values() for idx in group], rng))
            target_label = self.adata.obs.iloc[target_idx]["genotype"]
        else:
            target_label = self._choice(list(target_groups), rng)
            target_idx = int(self._choice(target_groups[target_label], rng))
        if target_label == self.control_value:
            return self._identity_pair(source_idx)

        if self.ot is not None:
            target_idx = self.ot.sample_target(source_idx, target_label, rng)
            if target_idx is None:
                return self._identity_pair(source_idx)
        return PerturbationPair(int(source_idx), target_idx, target_label, "perturbation")

    def _resolve_target_pair(self, target_idx, rng=None):
        target = self.adata.obs.iloc[target_idx]
        if target["genotype"] == self.control_value:
            return self._identity_pair(target_idx)
        if self.ot is not None:
            source_idx = self.ot.sample_source(target_idx, rng)
        else:
            source_pool = (self.source_cell_pool or {}).get(target["celltype"], [])
            if not source_pool:
                raise ValueError(f"No eligible sources available for context {target['celltype']!r}")
            source_idx = int(self._choice(source_pool, rng))
        return PerturbationPair(source_idx, int(target_idx), target["genotype"], "perturbation")

    def _resolve_anchor(self, anchor_idx, rng=None):
        if self.pairing_anchor == "target":
            return self._resolve_target_pair(anchor_idx, rng)
        return self._resolve_source_pair(anchor_idx, rng)

    def _build_fixed_pairs(self):
        rng = np.random.default_rng(self.seed)
        return tuple(self._resolve_anchor(anchor_idx, rng) for anchor_idx in self.anchor_indices)

    def __len__(self):
        return len(self.anchor_indices)

    def resolve(self, position):
        if self._fixed_pairs is not None:
            return self._fixed_pairs[position]
        return self._resolve_anchor(self.anchor_indices[position])

    @property
    def pairs(self):
        return self._fixed_pairs
