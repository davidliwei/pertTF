from typing import List, Tuple, Dict, Union, Optional, Literal
import time
import copy
from operator import itemgetter


from sklearn.model_selection import train_test_split, KFold
import torch
import numpy as np
import random
from scipy.sparse import issparse
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from anndata import AnnData
import anndata
from scipy.sparse import issparse
#from torchtext.vocab import Vocab
#from torchtext._torchtext import (
#    Vocab as VocabPybind,
#)
from sklearn.model_selection._split import _BaseKFold
from perttf.utils.custom_tokenizer import tokenize_and_pad_batch, random_mask_value, SimpleVocab


def add_batch_info(adata):
    """helper function to add batch effect columns into adata"""
    if "batch" not in adata.obs.columns: 
        batch_ids_0=random.choices( [0,1], k=adata.shape[0])
        adata.obs["batch"]=batch_ids_0
    if "batch_id" not in adata.obs.columns: 
        adata.obs["str_batch"] = adata.obs["batch"]
        adata.obs["str_batch"] = adata.obs["str_batch"].astype(str)
        adata.obs["batch_id"] = adata.obs["str_batch"].astype("category").cat.codes.values

"""
STEPS TO TRAIN:
create a PertTFDataManager First and use it generate loaders, validation and data_gen dictionary
Pass train_loader, valid_loader and data_gen to the wrapper_train either once or as part of kfold loop

"""

class PertTFDataset(Dataset):
    """
    A PyTorch Dataset for AnnData objects that performs next-cell sampling on the fly.
    """
    def __init__(self, adata: AnnData, indices: np.ndarray = None, 
                 cell_type_to_index: dict = None, genotype_to_index: dict = None, expr_layer: str = 'X_binned',
                 ps_columns: list = None, ps_columns_perturbed_genes: list = None, next_cell_pred: str = "identity", 
                 additional_ps_dict: dict = None, 
                 only_sample_wt_pert: bool = False):
        """
        The PertTFDataset serves to interface with pytorch Dataloaders 
        Its main function is to subset and extract single samples from a single Anndata object that is in-memory
        customized with the ability to sample a Perturbed sample for "WT" samples.

        Args:
            adata (AnnData): The full AnnData object, may be shared by multiple objects.
            indices (np.ndarray): The indices of the adata object that belong to this dataset (e.g., train or valid).
            config (object): A configuration object with parameters like 'binned_layer_key'.
            cell_type_to_index (dict): Mapping from cell type string to integer index.
            genotype_to_index (dict): Mapping from genotype string to integer index.
            ps_columns (list, optional): List of columns in obs to use for 'ps' scores.
            ps_columns_perturbed_genes (list, optional): List of perturbed genes for ps_columns. Only active if next_cell_red is "lochness". Have to be have the same length as ps_columns.
            next_cell_pred (str): The mode for next cell prediction ("identity" or "pert" or "lochness").
            additional_ps_dict (dict): a dictionary {gene_name:ps} the pass the additional ps score for other genes to the training.
            only_sample_wt_pert: only random sample next cell for wild type cells
        """
        self.adata = adata
        self._check_anndata_content()
        self.indices = indices if indices is not None else np.arange(len(self.adata.obs.index))
        self.expr_layer = expr_layer
        self.next_cell_pred = next_cell_pred
        
        # Mappings
        self.cell_type_to_index = cell_type_to_index if cell_type_to_index is not None else {t: i for i, t in enumerate(self.adata.obs['celltype'].unique())}
        self.genotype_to_index = genotype_to_index if genotype_to_index is not None else {t: i for i, t in enumerate(self.adata.obs['genotype'].unique())}
        self.ps_columns = ps_columns or [] 
        self.ps_columns = list(self.ps_columns) # must be a list 
        # For efficient next-cell sampling, pre-compute a dictionary of valid choices
        # IMPORTANT: This dictionary only contains cells from the current data split (train/valid)
        # to prevent data leakage.
        self.next_cell_dict = self._create_next_cell_pool()
        self.only_sample_wt_pert = only_sample_wt_pert
        if self.next_cell_pred == "lochness":
            if ps_columns is None:
                raise ValueError("PS columns must be provided for lochness prediction")
            if len(ps_columns) != len(ps_columns_perturbed_genes):
                raise ValueError("The ps_columns_perturbed_genes must be specified and has to have equal length as ps_columns")
            #perturbation_labels_uniq = np.unique(perturbation_labels_0)
            ps_columns_perturbed_genes = [x for x in ps_columns_perturbed_genes if x in self.genotype_to_index]
            if len(ps_columns_perturbed_genes) != len(ps_columns):
                print('Specified perturbed genes for PS column after filtering:' + ','.join(ps_columns_perturbed_genes))
                raise ValueError("The ps_columns_perturbed_genes must be specified and has to have equal length as ps_matrix")
            self.ps_columns_perturbed_genes = ps_columns_perturbed_genes

            if additional_ps_dict is None:
                self.additional_ps_dict={}
                self.additional_ps_names=[]
            else:
                self.additional_ps_dict = additional_ps_dict
                self.additional_ps_names = list(additional_ps_dict.keys())
                # sanity check: whether the gene keys are part of the genotype_to_index
                for x in self.additional_ps_names:
                    if x not in self.genotype_to_index:
                        raise ValueError(f"The gene name {x} provided in additional_ps_dict is not included in the pre-defined genotype.")

        # store the ps_matrix as numpy array for fast retrival during training
        self.ps_matrix = self.adata.obs[self.ps_columns].values.astype(np.float32) if self.ps_columns else np.zeros((self.adata.shape[0],1), dtype=np.float32)


    def _check_anndata_content(self):
        assert 'genotype' in self.adata.obs.columns and 'celltype' in self.adata.obs.columns, 'no genotype or celltype column found in anndata'
        add_batch_info(self.adata)
        
    def set_new_indices(self, indices):
        self.indices = indices
        self.next_cell_dict = self._create_next_cell_pool()

    def get_adata_subset(self, next_cell_pred = 'identity'):
        assert next_cell_pred in ['pert', 'identity', "lochness"], 'next_cell_pred can only be identity or pert or lochness'

        adata_small = self.adata[self.indices,].copy()
        if next_cell_pred == "identity" :
            return adata_small
        elif next_cell_pred == "lochness":
            adata_small.obs['genotype_next'] = adata_small.obs['genotype']
            return adata_small
        else:
            #adata_small = self.adata[self.indices,].copy()
            next_cell_id_list = []
            next_pert_list = []
            next_cell_global_idx_list = []
            for i in self.indices:
                current_cell_idx = self.adata.obs.index[i]
                current_cell_celltype = self.adata.obs.at[current_cell_idx, 'celltype']
                current_cell_genotype = self.adata.obs.at[current_cell_idx, 'genotype']
                next_cell_id, next_pert_label_str = self._sample_next_cell(current_cell_idx, current_cell_celltype, current_cell_genotype)
                next_cell_id_list.append(next_cell_id)
                next_pert_list.append(next_pert_label_str)
                next_cell_global_idx_list.append(self.adata.obs.index.get_loc(next_cell_id))
            adata_small.obs['genotype_next'] = next_pert_list
            adata_small.obs['next_cell_id'] = next_cell_id_list
            adata_small.layers['next_expr'] = self.adata.layers[self.expr_layer][next_cell_global_idx_list]
        return adata_small

    def __len__(self):
        return len(self.indices)

    def _create_next_cell_pool(self):
        """Pre-computes a dictionary for fast sampling of the next cell."""
        if self.next_cell_pred != "pert":
            return None
            
        next_cell_dict = {}
        # Use only the subset of adata relevant to this dataset split
        obs_subset = self.adata.obs.iloc[self.indices]
        
        for cell_type in obs_subset['celltype'].unique():
            next_cell_dict[cell_type] = {}
            for genotype in obs_subset['genotype'].unique():
                # Find cells matching the criteria within the current split
                mask = (obs_subset['celltype'] == cell_type) & (obs_subset['genotype'] == genotype)
                included_cells_indices = obs_subset[mask].index.tolist()
                if included_cells_indices:
                    next_cell_dict[cell_type][genotype] = included_cells_indices
        return next_cell_dict

    def _sample_next_cell(self,  current_cell_idx, current_cell_celltype, current_cell_genotype):
        """Samples a 'next cell' for a given current cell."""
        #current_cell_id = current_cell_obs.name
        #current_cell_type = current_cell_obs['celltype']
        #current_genotype = current_cell_obs['genotype']

        current_cell_id = current_cell_idx
        current_cell_type = current_cell_celltype
        current_genotype = current_cell_genotype

        if self.next_cell_pred == "identity" or self.next_cell_pred == "lochness":
            return current_cell_id, current_genotype

        # Logic for perturbation prediction
        valid_genotypes = self.next_cell_dict.get(current_cell_type, {})
        if not valid_genotypes:
             return current_cell_id, current_genotype # Fallback

        if current_genotype == 'WT':
            # Randomly select a different genotype
            next_pert_value = random.choice(list(valid_genotypes.keys()))
        else:
            # For non-WT, the "next" state is the same perturbation
            next_pert_value = current_genotype

        if next_pert_value == current_genotype and self.only_sample_wt_pert:
            return current_cell_id, next_pert_value
        else:
            # Sample a random cell with the target cell type and genotype
            possible_next_cells = valid_genotypes.get(next_pert_value, [current_cell_id])
            next_cell_id = random.choice(possible_next_cells)
            return next_cell_id, next_pert_value
    
    def __getitem__(self, idx: int):
        """
        Retrieves one sample from the dataset. This is where on-the-fly processing happens.
        """

        # 1. Get the index for the current cell
        current_cell_global_idx = self.indices[idx]

        #current_cell_obs = self.adata.obs.iloc[current_cell_global_idx] # too slow

        current_cell_idx = self.adata.obs.index[current_cell_global_idx]
        current_cell_celltype = self.adata.obs.at[current_cell_idx, 'celltype']
        current_cell_genotype = self.adata.obs.at[current_cell_idx, 'genotype']
        current_cell_batch_label = self.adata.obs.at[current_cell_idx, 'batch_id']

        # 2. Get expression data for the current cell
        binned_layer_key = self.expr_layer
        curr_gene = self.adata.var.index
        current_expr = self.adata.layers[binned_layer_key][current_cell_global_idx]
        if issparse(current_expr):
            current_expr = current_expr.toarray().flatten()

        # 3. Sample the next cell and its metadata
        next_cell_id, next_pert_label_str = self._sample_next_cell(current_cell_idx, current_cell_celltype,  current_cell_genotype)
        next_cell_global_idx = self.adata.obs.index.get_loc(next_cell_id)
        
        # 4. Get expression data for the next cell
        next_expr = self.adata.layers[binned_layer_key][next_cell_global_idx]

        next_gene = self.adata.var.index

        if issparse(next_expr):
            next_expr = next_expr.toarray().flatten()

        # 5. Get labels and PS scores
        cell_label = self.cell_type_to_index[current_cell_celltype]
        pert_label = self.genotype_to_index[current_cell_genotype]
        

        batch_label = current_cell_batch_label

        # Next cell labels are the same for cell type, but perturbation can change
        cell_label_next = cell_label
        pert_label_next = self.genotype_to_index[next_pert_label_str]

                
        #ps_scores = np.array([self.adata.obs.at[current_cell_idx, ps] for ps in self.ps_columns]).astype(np.float32) if self.ps_columns else np.array([0.0], dtype=np.float32)
        ps_scores = self.ps_matrix[current_cell_global_idx]
        #ps_scores_next = self.adata.obs.loc[next_cell_id, self.ps_columns].values.astype(np.float32) if self.ps_columns else np.array([0.0], dtype=np.float32)
        ps_scores_next = self.ps_matrix[next_cell_global_idx]
        if self.next_cell_pred == "lochness":
            #pert_label = self.genotype_to_index[current_cell_obs['genotype']]
            selection_pool_length = len(self.ps_columns_perturbed_genes) + len(self.additional_ps_names)
            random_pert_ind = random.randint(0, selection_pool_length-1)
            if random_pert_ind < len(self.ps_columns_perturbed_genes): # falls within the provided ps
                pert_label_next = self.genotype_to_index[self.ps_columns_perturbed_genes[random_pert_ind]] # this is the randomly assigned perturbations
                ps_scores_next = np.array([ps_scores[random_pert_ind]], dtype=np.float32)  # note that the target prediction is not the PS of NEXT cell, but the current cell
            else: # falls within additional ps scores defined in additional_ps_dict
                selected_gene = self.additional_ps_names[random_pert_ind - len(self.ps_columns_perturbed_genes)] 
                pert_label_next = self.genotype_to_index[selected_gene]
                ps_scores_next = np.array([self.additional_ps_dict[selected_gene]], dtype=np.float32) 
        
        return {
            "expr": current_expr,
            "expr_next": next_expr,
            "genes": curr_gene,
            "next_genes": next_gene,
            "celltype_labels": cell_label,
            "perturbation_labels": pert_label,
            "batch_labels": batch_label,
            "celltype_labels_next": cell_label_next,
            "perturbation_labels_next": pert_label_next,
            "ps": ps_scores,
            "ps_next": ps_scores_next,
            "index": current_cell_global_idx,
            "next_index": next_cell_global_idx,
            'name': current_cell_idx,
            'next_name': next_cell_id
        }


class PertBatchCollator:
    """
    A collate function for the DataLoader that tokenizes, pads, and masks batches on the fly.
    """
    def __init__(self, vocab: object, gene_ids: np.ndarray, full_tokenize: bool = False, **config):
        self.config = config
        self.vocab = vocab
        self.gene_ids = gene_ids
        self.full_tokenize = full_tokenize
        self.include_zero_gene = config.get('include_zero_gene', True)
        self.append_cls = config.get('append_cls', True)
        self.cls_value = config.get('cls_value', -3)
        self.cls_token = config.get('cls_token', '<cls>')
        self.max_seq_len = config.get('max_seq_len', 3000)
        self.pad_token = config.get('pad_token', '<pad>')
        self.pad_value = config.get('pad_value', -2)
        self.mask_ratio = config.get('mask_ratio', 0.15)
        self.mask_value = config.get('mask_value', -1)

    def __call__(self, batch: list) -> dict:
        """
        Processes a list of samples from the Dataset into a single batch tensor.
        """

        # Define which items to get from each dictionary
        get_values = itemgetter('expr', 'expr_next', 'genes', 'next_genes')
        # Create an iterator of tuples, then unzip them into four separate lists
        expr_list, expr_next_list, gene_list, gene_next_list = map(list, zip(*(get_values(item) for item in batch)))

        # 2. Tokenize and pad the expression data for the current batch
        # max seq len determines the context window for pertTF transformer modeling
        # during validation and predictions, this window may be around all genes with expression
        max_seq_len = self.max_seq_len if not self.full_tokenize else len(self.gene_ids) + self.append_cls

        # TODO: These functions may need to be modified to accomodate inputs w differing number of genes in the future
        tokenized, gene_idx_list = tokenize_and_pad_batch(
            np.array(expr_list), self.gene_ids, max_len=max_seq_len,cls_token=self.cls_token,
            vocab=self.vocab, pad_token=self.pad_token, pad_value=self.pad_value,
            append_cls=self.append_cls, include_zero_gene=self.include_zero_gene, 
            cls_value=self.cls_value
        )
        tokenized_next, _ = tokenize_and_pad_batch(
            np.array(expr_next_list), self.gene_ids, max_len=max_seq_len, cls_token=self.cls_token,
            vocab=self.vocab, pad_token=self.pad_token, pad_value=self.pad_value,
            append_cls=self.append_cls, include_zero_gene=self.include_zero_gene, 
            sample_indices=gene_idx_list, cls_value=self.cls_value
        )
        
        # 3. Apply random masking for this batch
        masked_values = random_mask_value(
            tokenized["values"], mask_ratio=self.mask_ratio,
            mask_value=self.mask_value, pad_value=self.pad_value,
            cls_value= self.cls_value
        )

        # 4. Collate all other labels into tensors
        collated_batch = {
            "gene_ids": tokenized["genes"],
            "next_gene_ids": tokenized_next["genes"],
            "values": masked_values,
            "target_values": tokenized["values"],
            "target_values_next": tokenized_next["values"],
        }
        
        # Stack scalar or vector labels from each item in the batch
        for key in batch[0].keys():
            if 'name' in key:
                values = [item[key] for item in batch]
                collated_batch[key] = values
            elif key not in ["expr", "expr_next"] and key not in ['genes', 'next_genes']:
                values = [item[key] for item in batch]
                tensor = torch.from_numpy(np.array(values))
                # Ensure labels are long type and scores are float
                collated_batch[key] = tensor.long() if 'label' in key else tensor.float()

        return collated_batch
    
    

class PertTFUniDataManager:
    """
    Manages data loading, preprocessing, and splitting using a single (Uni) AnnData object.
    This class encapsulates all data-related setup, including vocab, mappings,
    and provides methods to get data loaders for training and cross-validation.
    """
    def __init__(self, 
                 adata: AnnData, 
                 config: object, 
                 ps_columns: list = None,
                 ps_columns_perturbed_genes: list = None,
                 celltype_to_index: dict = None, 
                 vocab: SimpleVocab= None,
                 genotype_to_index: dict = None, 
                 expr_layer: str = 'X_binned',
                 next_cell_pred_type: str = "identity", 
                 additional_ps_dict: dict = None, 
                 only_sample_wt_pert: bool = False):
        #assert not adata.is_view, "The provided anndata is likely a view of the original anndata, this is probably due to slicing the original annadata object, please use the .copy() method to provide a copy"
        self.adata = adata.copy() # make a copy of the data so that no issues arise if adata is a anndata view
        self.indices = np.arange(self.adata.n_obs)
        self.config = config
        self.ps_columns = ps_columns # perhaps this can incorporated into config
        self.ps_columns_perturbed_genes = ps_columns_perturbed_genes
        self.additional_ps_dict = additional_ps_dict
        self.expr_layer = expr_layer
        self.only_sample_wt_pert = config.get('only_sample_wt_pert', only_sample_wt_pert)
        self.next_cell_pred_type = config.get('next_cell_pred_type', next_cell_pred_type)
        # --- Perform one-time data setup ---
        print("Initializing PertTFUniDataManager: Creating vocab and mappings...")
        #if "batch_id" not in self.adata.obs.columns:
         #   self.adata.obs["str_batch"] = "batch_0"
          #  self.adata.obs["batch_id"] = self.adata.obs["str_batch"].astype("category").cat.codes
        
        # Create and store mappings and vocab as instance attributes
                #self.num_batch_types = len(self.adata.obs["batch_id"].unique())

        
        self.genes = self.adata.var.index.tolist()
        self.vocab = SimpleVocab(self.genes, config.special_tokens) if vocab is None else vocab
        assert self.adata.var.index.isin(self.vocab.stoi).all(), 'Not all genes are in provided vocab, please prefiltered the Anndata first'
        self.vocab.set_default_index(self.vocab["<pad>"])
        self.gene_ids = np.array(self.vocab(self.genes), dtype=int)

        self.set_genotype_index(genotype_to_index= genotype_to_index)
        self.set_celltype_index(celltype_to_index= celltype_to_index)
        
        add_batch_info(self.adata)
        self.num_batch_types = len(self.adata.obs["batch_id"].unique())
        # The collators can be created once and reused
        ## first collator is the training collator, with a context window set in config
        self.collator = PertBatchCollator(self.vocab, self.gene_ids, **config)
        ## full collator may be used for validation or inference 
        ## This may be very slow for full gene set, scaling is roughly 2x context length -> 3.6x time, 3-4x more memory
        self.full_token_collator = PertBatchCollator( self.vocab, self.gene_ids, full_tokenize=True, **config)
        print("Initialization complete.")

    def set_genotype_index(self, genotype_to_index):
        self.genotype_to_index = {t: i for i, t in enumerate(self.adata.obs['genotype'].unique())} if genotype_to_index is None else genotype_to_index
        self.num_genotypes = len(self.genotype_to_index)

    def set_celltype_index(self, celltype_to_index):
        self.cell_type_to_index = {t: i for i, t in enumerate(self.adata.obs['celltype'].unique())} if celltype_to_index is None else celltype_to_index
        self.num_cell_types = len(self.cell_type_to_index)

    def get_adata_info_dict(self):
        data_gen = { 
            'genes': self.genes,
            'gene_ids': self.gene_ids,
            'vocab': self.vocab,
            'num_batch_types': self.num_batch_types, # need to change this
            'num_cell_types': self.num_cell_types,
            'num_genotypes': self.num_genotypes,
            'cell_type_to_index': self.cell_type_to_index,
            'genotype_to_index': self.genotype_to_index
        }
        if self.ps_columns is not None:
            data_gen['ps_names']=[x for x in self.ps_columns if x in self.adata.obs.columns]
        else:
            data_gen['ps_names']=["PS"]
                
        return data_gen

    def _create_dataset_from_indices(self, indices):
        """A helper function to create PertTFDataset from underlying adata."""
        perttf_dataset = PertTFDataset(
            self.adata, indices=indices, cell_type_to_index=self.cell_type_to_index, genotype_to_index=self.genotype_to_index,
            ps_columns=self.ps_columns, ps_columns_perturbed_genes = self.ps_columns_perturbed_genes, 
            next_cell_pred=self.next_cell_pred_type ,    additional_ps_dict = self.additional_ps_dict,  
            expr_layer=self.expr_layer, only_sample_wt_pert=self.only_sample_wt_pert
        )
        return perttf_dataset

    def _create_loaders_from_dataset(self, dataset, full_token_collator = False):
        """A helper function to create dataloaders from PertTFDataset."""    
        collator = self.collator if not full_token_collator else self.full_token_collator 
        loader = DataLoader(
            dataset, batch_size=self.config.batch_size, shuffle=True,
            num_workers=8, collate_fn=collator, pin_memory=True
        )
        return loader

    def get_data_w_loader(self, indices = None, full_data = False, full_token = False):
        indices = self.indices if indices is None and full_data else indices
        data = self._create_dataset_from_indices(indices)
        loader = self._create_loaders_from_dataset(data, full_token)
        return data, loader

    def get_train_valid_loaders(self, test_size: float = 0.1, train_indices = None, valid_indices = None, full_token_validate  = False):
        """Provides a single, standard train/validation split."""
        print(f"Creating a single train/validation split (test_size={test_size})...")
        if train_indices is None or valid_indices is None:
            indices = np.arange(self.adata.n_obs)
            train_indices, valid_indices = train_test_split(indices, test_size=test_size, shuffle=True)
        else:
            assert len(set(train_indices).intersection(valid_indices)) == 0, 'training data and validation data are not separate'
            print('overiding random train/valid split with provided indices')
        train_data, train_loader = self.get_data_w_loader(train_indices)
        valid_data, valid_loader = self.get_data_w_loader(valid_indices, full_token=full_token_validate)
        return train_data, train_loader, valid_data, valid_loader, self.get_adata_info_dict()

    def get_k_fold_split_loaders(self, cv = 5):
        """
        An iterator that yields train and validation dataloaders for each fold
        in a k-fold cross-validation setup.
        """
        kf = cv if issubclass(cv.__class__,  _BaseKFold) else KFold(n_splits=cv, shuffle=True)
        print(f"Set up K-Fold cross-validation with {kf.n_splits} folds")
        for fold, (train_indices, valid_indices) in enumerate(kf.split(self.indices)):
            print(f"--- Yielding data loaders for Fold {fold+1}/{kf.n_splits} ---")
            yield self.get_train_valid_loaders(train_indices = train_indices, valid_indices = valid_indices, full_token_validate  = False)



"""
The following classes attempt to implement a crude solution for dealing with large amount of single cell datasets
The General Idea is as follows:
1. Massive datasets should be processed into relatively unfied format (prefer Anndata) and placed on disk.
1. Create Manifest that contains the meta data of all cells and raw files of each cell
2. Train/valid split based on Manifest
3. Memory constraints are shifted to I/O via on the fly read/loading of each cell with Anndata backed='r' or equivalent
4. Collators and other preprocessing functions should remain unchanged
5. This should also accomodate the case of a single massive dataset
"""

import pandas as pd
import anndata

class PertTFMultiDataManager:
    """Unfinished Implementation"""
    def __init__(self, data_directory: str, config: object):
        self.config = config
        self.data_directory = data_directory
        
        # 1. Create or load the global manifest
        self.manifest_path = "global_manifest.parquet"
        if not os.path.exists(self.manifest_path):
            print("Creating global manifest...")
            self.manifest = self._create_manifest()
        else:
            print("Loading existing manifest...")
            self.manifest = pd.read_parquet(self.manifest_path)

        # 2. Create global mappings from the manifest
        print("Creating global vocab and mappings...")
        self.cell_type_to_index = {t: i for i, t in enumerate(self.manifest['celltype'].unique())}
        self.genotype_to_index = {t: i for i, t in enumerate(self.manifest['genotype'].unique())}
        # Assume genes are consistent across files, or create a global gene list
        # by inspecting the first file.
        first_adata = anndata.read_h5ad(self.manifest['file_path'].iloc[0])
        self.genes = first_adata.var.index.tolist()
        self.vocab = self._create_vocab()
        self.gene_ids = np.array(self.vocab(self.genes), dtype=int)

        # Collators can remain largely the same!
        self.collator = PertBatchCollator(...)
        
    def _create_manifest(self):
        all_obs = []
        h5ad_files = glob.glob(f"{self.data_directory}/*.h5ad")
        for file_path in h5ad_files:
            adata = anndata.read_h5ad(file_path, backed='r')
            obs_df = adata.obs.copy()
            obs_df['file_path'] = file_path
            obs_df['index_in_file'] = np.arange(adata.n_obs)
            all_obs.append(obs_df)
        
        manifest_df = pd.concat(all_obs)
        manifest_df.to_parquet(self.manifest_path)
        return manifest_df

    # ... other methods ...


class PertTFMultiDataset(Dataset):
    """Unifinished Implementation"""
    def __init__(self, manifest_df: pd.DataFrame, cell_type_to_index: dict, genotype_to_index: dict):
        """
        Initializes with a manifest DataFrame for a specific data split (e.g., train or valid).
        """
        self.manifest = manifest_df.reset_index(drop=True)
        self.cell_type_to_index = cell_type_to_index
        self.genotype_to_index = genotype_to_index
        self.expr_layer = 'X_binned'
        # ... other configs ...

        # The sampling pool is now built from the manifest, not an AnnData object.
        # This is fast and memory-efficient.
        self.next_cell_dict = self._create_next_cell_pool()

    def __len__(self):
        return len(self.manifest)

    def _create_next_cell_pool(self):
        """Pre-computes a dictionary for sampling using the manifest."""
        next_cell_dict = {}
        # Group by celltype and genotype to find valid indices within the manifest
        for (cell_type, genotype), group in self.manifest.groupby(['celltype', 'genotype']):
            if cell_type not in next_cell_dict:
                next_cell_dict[cell_type] = {}
            # Store the manifest indices (0, 1, 2...) for this group
            next_cell_dict[cell_type][genotype] = group.index.tolist()
        return next_cell_dict

    def _get_cell_data(self, manifest_idx: int):
        """Helper to load data for a single cell from disk using its manifest index."""
        cell_info = self.manifest.iloc[manifest_idx]
        file_path = cell_info['file_path']
        index_in_file = cell_info['index_in_file']
        
        # Use backed mode for memory-efficient reading of a single row
        adata_slice = anndata.read_h5ad(file_path, backed='r')
        expr = adata_slice.layers[self.expr_layer][index_in_file]
        if issparse(expr):
            expr = expr.toarray().flatten()
            
        return expr, cell_info

    def __getitem__(self, idx: int):
        """
        Retrieves one sample by reading from disk on the fly.
        """
        # 1. Get current cell's metadata and expression data
        current_expr, current_cell_info = self._get_cell_data(idx)

        # 2. Sample the next cell using the manifest-based pool
        # This logic remains very similar, but now returns a manifest index
        next_cell_manifest_idx, next_pert_label_str = self._sample_next_cell(current_cell_info)
        
        # 3. Get next cell's expression data
        next_expr, next_cell_info = self._get_cell_data(next_cell_manifest_idx)
        
        # 4. Assemble the sample dictionary (similar to your original code)
        # ... use current_cell_info and next_cell_info to get labels ...
        sample = {
            "expr": current_expr,
            "expr_next": next_expr,
            # ... other key-value pairs
        }
        return sample

    def _sample_next_cell(self, current_cell_info):
        """Samples a 'next cell' using the manifest index."""
        # This logic is adapted to use the manifest and its indices
        current_cell_type = current_cell_info['celltype']
        current_genotype = current_cell_info['genotype']
        current_manifest_idx = current_cell_info.name # .name holds the manifest index

        # ... (sampling logic similar to before, but now chooses a manifest index
        # from self.next_cell_dict) ...

        # For example:
        #possible_next_indices = self.next_cell_dict[target_cell_type][target_genotype]
        #next_manifest_idx = random.choice(possible_next_indices)
        
        #return next_manifest_idx, target_genotype
