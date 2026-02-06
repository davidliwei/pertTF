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
                              logger = None):
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
    """
    test_manager = PertTFUniDataManager(adata_input, 
                                     config, 
                                     ps_columns=ps_columns, vocab=vocab,
                                     ps_columns_perturbed_genes=ps_columns_perturbed_genes, 
                                     additional_ps_dict = additional_ps_dict,
                                     celltype_to_index = cell_type_to_index, 
                                     genotype_to_index= genotype_to_index, 
                                     expr_layer= input_layer_key)
    random_state = config.get('seed', None)
    t_data, t_loader, v_data, v_loader, data_info = test_manager.get_train_valid_loaders(test_size=train_val_split, train_indices=train_indices, valid_indices=valid_indices, full_token_validate=full_token_validate, random_state = random_state)
    data_info['train_loader'] = t_loader
    data_info['valid_loader'] = v_loader
    data_info['train_data'] = t_data
    data_info['valid_data'] = v_data
    data_info['cell_ids_train'] = t_data.get_adata_subset().obs.index
    data_info['adata_sorted'] = v_data.get_adata_subset(next_cell_pred=next_cell_pred)
    data_info['adata_manager'] = test_manager
    data_info['n_perturb'] = data_info['num_genotypes']
    data_info['n_cls'] = data_info['num_cell_types']
    return data_info
