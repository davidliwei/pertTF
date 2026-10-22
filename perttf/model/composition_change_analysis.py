import warnings

import torch
import scanpy as sc
import numpy as np
import pandas as pd
from scipy import sparse
from perttf.model.train_function import eval_testdata
from perttf.utils.pert_labels import split_perturbation_label

def _carrier_matrix(labels, pert_delimiter=None, control_value="WT"):
    """Sparse cells x groups 0/1 membership. Groups are whole labels when ``pert_delimiter`` is None,
    otherwise perturbation components (a cell labelled "A+B" carries both A and B)."""
    labels = np.asarray(labels).astype(str)
    uniq, inverse = np.unique(labels, return_inverse=True)
    if pert_delimiter is None:
        components = [(label,) for label in uniq]
    else:
        components = [split_perturbation_label(label, pert_delimiter, control_value) for label in uniq]
    groups = sorted({c for comps in components for c in comps})
    index = {g: j for j, g in enumerate(groups)}
    rows = [i for i, comps in enumerate(components) for _ in comps]
    cols = [index[c] for comps in components for c in comps]
    label_to_group = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(uniq), len(groups)))
    return label_to_group[inverse.ravel()].tocsr(), groups


def _neighbor_indicator(adata, nn_key):
    """0/1 kNN adjacency from a distances matrix; zero distances (the cell itself) are dropped."""
    adjacency = sparse.csr_matrix(adata.obsp[nn_key], dtype=np.float64, copy=True)
    adjacency.data = (adjacency.data != 0).astype(np.float64)
    adjacency.eliminate_zeros()
    return adjacency


def _carrier_neighbor_counts(adjacency, membership, chunk_size=4096):
    """Per cell and group: number of the cell's neighbours carrying the group, restricted to the groups
    the cell itself carries (rows chunked; at most n_neighbors x perturbations-per-cell entries per row)."""
    out = []
    for start in range(0, adjacency.shape[0], chunk_size):
        rows = slice(start, start + chunk_size)
        out.append((adjacency[rows] @ membership).multiply(membership[rows]))
    return sparse.vstack(out).tocsr()


def _group_fractions(groups, membership, overall_fraction_dict):
    if overall_fraction_dict is None:
        return np.asarray(membership.sum(axis=0)).ravel() / membership.shape[0]
    missing = [g for g in groups if g not in overall_fraction_dict]
    if missing:
        raise ValueError(f"overall_fraction_dict has no entry for: {missing[:10]}")
    fractions = np.array([overall_fraction_dict[g] for g in groups], dtype=np.float64)
    fractions[fractions == 0] = 0.0001
    return fractions


def calculate_lonESS_score(adata, overall_fraction_dict=None,
                           recalculate_nn=False,
                           nn_name=None,
                           n_neighbors=30,
                           target_genotype=None,
                           n_pcs=20,
                           delta=0.0001,
                           pert_delimiter=None,
                           control_value="WT",
                           min_expected_neighbors=1.0,
                           ):
    """Per-cell lochNESS score: local enrichment of a perturbation among a cell's kNN neighbours.

    score = (neighbours carrying g) / n_neighbors / (overall fraction of cells carrying g) - 1, where g is
    ``target_genotype`` for every cell, or each cell's own genotype when ``target_genotype`` is None.
    Optional Gaussian noise of SD ``delta`` is added (ties breaker for training targets).

    Multi-perturbation labels: with ``pert_delimiter`` (e.g. "+"), "carrying g" means g is a component
    of the cell's label, ``target_genotype`` must be a single component, and the own-genotype score of a
    cell with several components is the mean of its per-component scores. Without ``pert_delimiter``
    labels are matched whole (the original behaviour, identical results).

    Warns when the expected number of carrier neighbours (n_neighbors x carrier fraction) is below
    ``min_expected_neighbors`` (None disables): the per-cell score is then dominated by counting noise,
    as with genome-wide high-MOI screens (~0.1-0.2 expected carriers among 300 neighbours).

    Returns a list with one score per cell (in ``adata.obs_names`` order).
    """
    if recalculate_nn:
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, key_added=nn_name)
    nn_key = 'distances' if nn_name is None else f'{nn_name}_distances'
    membership, groups = _carrier_matrix(adata.obs['genotype'].to_numpy(), pert_delimiter, control_value)
    adjacency = _neighbor_indicator(adata, nn_key)

    if target_genotype is not None:
        target = str(target_genotype)
        if pert_delimiter is not None and pert_delimiter in target:
            raise ValueError(f"target_genotype must be a single perturbation with pert_delimiter set, got {target!r}")
        if target in groups:
            j = groups.index(target)
            counts = np.asarray(adjacency @ membership[:, j].toarray()).ravel()
            fraction = _group_fractions([target], membership[:, [j]], overall_fraction_dict)[0]
        elif overall_fraction_dict is not None and target in overall_fraction_dict:
            counts = np.zeros(adata.n_obs)
            fraction = _group_fractions([target], None, overall_fraction_dict)[0]
        else:
            hint = "" if pert_delimiter is not None else " (multi-perturbation labels need pert_delimiter)"
            raise ValueError(f"target_genotype {target!r} is not carried by any cell{hint}")
        scores = counts / n_neighbors / fraction - 1
        expected = n_neighbors * fraction
    else:
        fractions = _group_fractions(groups, membership, overall_fraction_dict)
        weighted = _carrier_neighbor_counts(adjacency, membership) @ (1.0 / fractions)
        n_components = np.asarray(membership.sum(axis=1)).ravel()
        scores = weighted / n_neighbors / n_components - 1
        expected = float(np.median((membership @ (n_neighbors * fractions)) / n_components))

    if min_expected_neighbors is not None and expected < min_expected_neighbors:
        warnings.warn(
            f"lochNESS: expected carrier neighbours per cell = {expected:.3g} (< {min_expected_neighbors}); "
            "the per-cell score is dominated by counting noise. Use more neighbours, pool perturbations, "
            "or summarise per perturbation (lochness_by_perturbation).",
            stacklevel=2,
        )
    if delta > 0:
        scores = scores + np.random.normal(0, delta, size=scores.size)
    return scores.tolist()


def lochness_by_perturbation(adata, nn_name=None, n_neighbors=30, pert_delimiter=None, control_value="WT",
                             min_cells=1):
    """Per-perturbation lochNESS summary: do the carriers of g sit next to each other?

    For each perturbation g carried by >= ``min_cells`` cells (components when ``pert_delimiter`` is set):
      n_carriers                  cells carrying g (m);
      expected_neighbors_per_cell n_neighbors x m / n: carrier neighbours a cell expects under random placement
                                  (counting noise dominates the per-cell score when this is << 1);
      observed                    total carriers of g among the neighbours of g's carriers;
      expected                    m x n_neighbors x (m - 1) / (n - 1), the same total under random placement;
      enrichment                  observed / expected - 1 (0 under random placement; approximately the
                                  carrier-mean of calculate_lonESS_score, which uses m / n instead).
    Pooling over carriers gives ~m x expected_neighbors_per_cell counts per perturbation, so the summary is
    measurable where the per-cell score is not. Cells carrying more perturbations can co-localise (e.g. if
    guide capture tracks cell state): compare small enrichments with size-matched random carrier sets.
    Uses the kNN graph in ``adata.obsp`` (``nn_name``), self excluded. Returns a DataFrame indexed by g.
    """
    nn_key = 'distances' if nn_name is None else f'{nn_name}_distances'
    membership, groups = _carrier_matrix(adata.obs['genotype'].to_numpy(), pert_delimiter, control_value)
    adjacency = _neighbor_indicator(adata, nn_key)
    n = adata.n_obs
    m = np.asarray(membership.sum(axis=0)).ravel()
    observed = np.asarray(_carrier_neighbor_counts(adjacency, membership).sum(axis=0)).ravel()
    expected = m * n_neighbors * (m - 1) / max(n - 1, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        enrichment = np.where(expected > 0, observed / expected - 1, np.nan)
    table = pd.DataFrame({
        "n_carriers": m.astype(int),
        "expected_neighbors_per_cell": n_neighbors * m / n,
        "observed": observed,
        "expected": expected,
        "enrichment": enrichment,
    }, index=pd.Index(groups, name="perturbation"))
    return table[table.n_carriers >= min_cells]


def generate_lochness_ranking( adata_wt, candidate_genes,
                             model, gene_ids, cell_type_to_index, genotype_to_index, vocab,
                             config, device,
                             n_expands_per_epoch = 50,
                             n_epoch = 4,
                             wt_pred_next_label = "WT", ):
    """
    Generate perturbation embeddings for target cells, and calculate cosine similarity between all target cells vs wild-type cells

    Args:
      adata_wt: AnnData object of the wildtype cells. This is the cell population where the lochness score is based on
      candidate_genes: List of candidate genes for perturbation.
      model:
      gene_ids:
      cell_type_to_index:
      genotype_to_index:
      vocab:
      config:
      device:
      n_expands_per_epoch: the number of duplicates in adata_target to be used for perturbation simulation. For smaller number of target cells, set it to a big number
      n_epoch:  the number of rounds that perturbaiton prediction is performed
      wt_pred_next_label: This should fill the "pred_next" label for adata_wt. Default WT
    Returns:
      cell_emb_data_all: generated cell embeddings, a 2-d np array. Row size: (adata_target.n_obs*n_expands_per_epoch + adata_wt.n_obs)*n_epoch. Column size: (emb_size of the model)
      perturb_info_all: a Pandas dataframe describing the cell information in cell_emb_data_all
      cs_matrix_res: cosine similarity matrix, a 2-d np array. Size: adata_wt.n_obs * (adata_target.n_obs*n_expands_per_epoch) * n_epoch
      a_eva: evaluated AnnData object from the last round of evaluation
    """
    # expand
    adata_bwmerge=sc.concat([adata_wt]*n_expands_per_epoch,axis=0,)
    adata_bwmerge.var=adata_wt.var
    #cell_emb_data_all = None
    perturb_info_all = None

    #cs_matrix_res = np.zeros((adata_wt.shape[0], adata_target.shape[0] * n_expands_per_epoch, n_epoch))
    # loop over epochs
    for n_round in range(n_epoch):
        # assign genoytpe_next
        gt_next_1 = np.random.choice(list(candidate_genes), size = adata_wt.shape[0] * n_expands_per_epoch)

        adata_bwmerge.obs[ 'genotype_next'] = gt_next_1

        # feed into model
        model.to(device)
        #import pdb; pdb.set_trace()
        eval_results_0 = eval_testdata(model, adata_bwmerge, gene_ids,
                                    train_data_dict={"cell_type_to_index":cell_type_to_index,
                                                      "genotype_to_index":genotype_to_index,
                                                      "vocab":vocab,},
                                    config = config,
                                    make_plots=False)
        #
        a_eva=eval_results_0 #['adata']
        pred_ps_score = a_eva.obsm['ps_pred_next']
        perturb_info=a_eva.obs[['genotype','genotype_next','celltype',]]

        perturb_info['round']=n_round
        # Check if pred_ps_score is 2D and select the first column if so
        if pred_ps_score.ndim == 2 and pred_ps_score.shape[1] > 1:
            print('WARNING: pred_ps_score is 2D, probably due to the model pred_lochness_next parameter not properly set up. Selecting the first column.')
            pred_ps_score_1d = np.asarray(pred_ps_score[:, 0]).ravel()
        else:
            pred_ps_score_1d = np.asarray(pred_ps_score).ravel()
        perturb_info['pred_ps'] = pred_ps_score_1d
        #perturb_info['type']=['pert_source']*adata_target.shape[0]*n_expands_per_epoch + ['pert_dest']*adata_wt.shape[0]

        if perturb_info_all is None:
            perturb_info_all = perturb_info

        else:
            perturb_info_all = pd.concat([perturb_info_all, perturb_info], axis=0)

    perturb_info_all.reset_index(inplace=True)
    return  perturb_info_all,  a_eva



def aggregate_lochness_scores(perturb_info_src):
    avg_lochness_fp = perturb_info_src.groupby('genotype_next')['pred_ps'].agg(
        mean='mean',
        median='median',
        count='count',
        percentile_90=lambda x: np.percentile(x, 90),
        percentile_10=lambda x: np.percentile(x, 10),)

    # Sort the result
    avg_lochness_fp = avg_lochness_fp.sort_values(by='mean',ascending=True)

    # Add ranking
    avg_lochness_fp = avg_lochness_fp.reset_index()
    avg_lochness_fp['rank'] = avg_lochness_fp['mean'].rank(ascending=True, method='dense')
    avg_lochness_fp['rank_median'] = avg_lochness_fp['median'].rank(ascending=True, method='dense')
    avg_lochness_fp['rank_percentile90'] = avg_lochness_fp['percentile_90'].rank(ascending=True, method='dense')
    avg_lochness_fp['rank_percentile10'] = avg_lochness_fp['percentile_10'].rank(ascending=True, method='dense')

    avg_lochness_fp.set_index('genotype_next', inplace=True)
    avg_lochness_fp = avg_lochness_fp.sort_values(by='median',ascending=True)
    return avg_lochness_fp


