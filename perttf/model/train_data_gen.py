from typing import List, Tuple, Dict, Union, Optional, Literal
import random
import torch
import numpy as np
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader

from scipy.sparse import issparse

from anndata import AnnData
from sklearn.model_selection import train_test_split

from ..utils.custom_tokenizer import tokenize_and_pad_batch, random_mask_value, SimpleVocab
from ..utils.pert_data_loader import PertBatchCollator, PertTFDataset, PertTFUniDataManager, add_batch_info
from ..utils.logger import create_logger
from ..utils.split_check import check_train_valid_split

def produce_training_datasets(adata_input, config,
                              input_layer_key = "X_binned",
                              next_layer_key = "X_binned_next",
                              next_cell_pred: Literal["identity","pert","lochness"] = "identity",
                              cell_type_to_index = None,
                              genotype_to_index = None,
                              vocab = None,
                              ps_columns = None,
                              ps_columns_perturbed_genes = None,
                              additional_ps_dict = None,
                              full_token_validate = False,
                              train_val_split = 0.2,
                              train_indices = None,
                              valid_indices = None,
                              logger = None,
                              stratify_by = None,
                              split_check_columns = None):
    """
    produce training datasets for from scRNA-seq 
    Args:

    adata_input (:class:`AnnData`):
        The :class:`AnnData` object to preprocess.
    input_layer_key (:class:`str`, optional):
        The key of :class:`AnnData.obs` to use for expression layer. Default is the binned expression layer.
    next_layer_key (:class:`str`, optional):
        The key of :class:`AnnData.obs` to use for next-stage expression layer. Default is the binned expression layer.
    next_cell_pred:
        Whether to generate next cell fate prediction. Default is "identity" (simply duplicating input_layer_key).
    train_indices, valid_indices:
        Complete train and validation cohorts. Both are required for perturbation prediction;
        identity and lochness generate a random split when both are omitted.
    """
    effective_next_cell_pred = config.get('next_cell_pred_type', next_cell_pred)
    random_state = config.get('seed', None)
    if (train_indices is None) != (valid_indices is None):
        raise ValueError("train_indices and valid_indices must either both be provided or both be omitted")
    if train_indices is None:
        if effective_next_cell_pred == "pert":
            raise ValueError("Perturbation training requires explicit train_indices and valid_indices")
        stratify = None
        if stratify_by is not None:
            stratify_columns = [stratify_by] if isinstance(stratify_by, str) else list(stratify_by)
            missing_columns = [column for column in stratify_columns if column not in adata_input.obs]
            if missing_columns:
                raise ValueError(f"AnnData is missing stratification columns: {missing_columns}")
            if adata_input.obs[stratify_columns].isna().any().any():
                raise ValueError("Stratification columns contain missing values")
            stratify = adata_input.obs[stratify_columns].astype(str).agg("|".join, axis=1).to_numpy()
        train_indices, valid_indices = train_test_split(
            np.arange(adata_input.n_obs),
            test_size=train_val_split,
            shuffle=True,
            random_state=random_state,
            stratify=stratify,
        )
    elif stratify_by is not None:
        raise ValueError("stratify_by cannot be used with explicit train_indices and valid_indices")

    control_value = (config.get('pairing_config', {}) or {}).get(
        'control_value', config.get('perturbation_control_value', 'WT')
    )
    split_check = check_train_valid_split(
        adata_input,
        train_indices,
        valid_indices,
        mode=effective_next_cell_pred,
        control_value=control_value,
        check_columns=split_check_columns,
        min_cells=config.get('perturbation_metric_min_cells', 30),
    )
    split_summary = {
        'mode': effective_next_cell_pred,
        'n_train': split_check['n_train'],
        'n_valid': split_check['n_valid'],
    }
    if effective_next_cell_pred == 'pert':
        split_summary['n_overlapping_controls'] = split_check['n_overlapping_controls']
        split_summary['combination_type_counts'] = split_check['combination_type_counts']
        split_summary['context_exposure_counts'] = {
            name: len(contexts) for name, contexts in split_check['context_exposure'].items()
        }
    if logger is None:
        print(f"Train/validation split check: {split_summary}")
    else:
        logger.info(f"Train/validation split check: {split_summary}")

    test_manager = PertTFUniDataManager(adata_input,
                                     config,
                                     ps_columns=ps_columns, vocab=vocab,
                                     ps_columns_perturbed_genes=ps_columns_perturbed_genes, 
                                     additional_ps_dict = additional_ps_dict,
                                     celltype_to_index = cell_type_to_index,
                                     genotype_to_index= genotype_to_index,
                                     expr_layer= input_layer_key)
    t_data, t_loader = test_manager.get_data_w_loader(train_indices)
    perturbation_validation = effective_next_cell_pred == "pert"
    valid_control_indices = np.array([], dtype=np.int64)
    valid_target_indices = np.array([], dtype=np.int64)
    if perturbation_validation:
        valid_indices = np.asarray(valid_indices, dtype=np.int64)
        valid_is_control = adata_input.obs.iloc[valid_indices]['genotype'].to_numpy() == control_value
        valid_control_indices = valid_indices[valid_is_control]
        valid_target_indices = valid_indices[~valid_is_control]
        v_data, v_loader = test_manager.get_data_w_loader(
            valid_target_indices,
            pairing_config={
                'pairing_anchor': 'target',
                'source_indices': valid_control_indices,
                'target_indices': valid_target_indices,
                'source_selection': 'strict',
                'target_selection': 'strict',
                'identity_condition': None,
                'pair_schedule': 'fixed',
                'target_sampling': 'cell',
                'pairing_seed': random_state,
            },
            shuffle=False,
            deterministic=True,
            seed=0 if random_state is None else random_state,
            use_ot=False,
        )
    else:
        v_data, v_loader = test_manager.get_data_w_loader(
            valid_indices,
            full_token=full_token_validate,
            shuffle=False,
        )

    data_info = test_manager.get_adata_info_dict()
    data_info['train_loader'] = t_loader
    data_info['valid_loader'] = v_loader
    data_info['train_data'] = t_data
    data_info['valid_data'] = v_data
    data_info['cell_ids_train'] = t_data.get_adata_subset().obs.index
    data_info['adata_sorted'] = v_data.get_adata_subset(next_cell_pred=effective_next_cell_pred)
    data_info['adata_manager'] = test_manager
    data_info['train_indices'] = np.asarray(train_indices, dtype=np.int64)
    data_info['valid_indices'] = np.asarray(valid_indices, dtype=np.int64)
    data_info['split_check'] = split_check
    if perturbation_validation:
        if v_data.pairs is None:
            raise RuntimeError("Fixed perturbation validation did not produce stored pairs")
        data_info['perturbation_validation'] = True
        data_info['pert_valid_pairs'] = np.asarray(
            [[pair.source_idx, pair.target_idx] for pair in v_data.pairs], dtype=np.int64
        )
        data_info['pert_valid_target_indices'] = valid_target_indices
        data_info['pert_valid_control_indices'] = valid_control_indices
    data_info['n_perturb'] = data_info['num_genotypes']
    data_info['n_cls'] = data_info['num_cell_types']
    return data_info
