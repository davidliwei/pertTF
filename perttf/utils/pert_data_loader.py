from typing import List, Tuple, Dict, Union, Optional, Literal
from sklearn.model_selection import train_test_split, KFold
import torch
import numpy as np
import random
from scipy.sparse import issparse
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from anndata import AnnData
from scipy.sparse import issparse
from sklearn.model_selection._split import _BaseKFold
from .custom_tokenizer import tokenize_and_pad_batch, random_mask_value, SimpleVocab
from .misc import _get_sf
from .pert_pairing import ContextAwarePairing

# add batch info 
def add_batch_info(adata):
    """helper function to add batch effect columns into adata"""
    if "batch" not in adata.obs.columns: 
        batch_ids_0=random.choices( [0], k=adata.shape[0])
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
    def __init__(self, 
                 adata: AnnData, 
                 indices: np.ndarray = None, 
                 # OT Parameters
                 use_ot: bool = False,
                 ot_params: dict = dict(
                    ot_pickle_path= None,
                    ot_top_k= 10,
                    ot_epsilon = "auto",
                    ot_max_dist = "auto",
                    ot_epsilon_scaler= 0.01
                    ),
                 # Standard Parameters
                 cell_type_to_index: dict = None, 
                 genotype_to_index: dict = None, 
                 expr_layer: str = 'X_binned',
                 ps_columns: list = None, 
                 ps_columns_perturbed_genes: list = None, 
                 next_cell_pred: str = "identity", 
                 additional_ps_dict: dict = None, 
                 only_sample_wt_pert: bool = False,
                 size_factor_col: str = None,
                 prediction_only: bool = False,
                 pairing_anchor: Literal["source", "target"] = "source",
                 source_indices: Optional[np.ndarray] = None,
                 target_indices: Optional[np.ndarray] = None,
                 source_selection: Literal["strict", "all"] = "all",
                 target_selection: Literal["strict", "all"] = "all",
                 identity_condition: Optional[Literal["source"]] = "source",
                 pair_schedule: Literal["dynamic", "fixed"] = "dynamic",
                 target_sampling: Literal["cell", "perturbation"] = "perturbation",
                 pairing_seed: Optional[int] = None,
                 control_value: str = "WT",
                 ):
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
            only_sample_wt_pert: Legacy argument retained for callers; identity pairs now always reuse the source row.
            pairing_anchor: Whether each dataset row starts from a source or target index.
            source_selection: "strict" keeps only controls; "all" keeps all eligible sources.
            target_selection: "strict" keeps only perturbed targets; "all" keeps all eligible targets.
            identity_condition: "source" uses the source genotype; None uses the control condition.
            pair_schedule: Resolve pairs once with "fixed" or per item with "dynamic".
            target_sampling: Sample target cells directly or sample a perturbation before its target cell.
        """
        self.adata = adata
        self._check_anndata_content()
        indices = np.asarray(indices if indices is not None else np.arange(self.adata.n_obs), dtype=np.int64)
        self.next_cell_pred = next_cell_pred
        self.prediction_only = prediction_only
        self.use_ot = use_ot
        self.ot_params = ot_params
        self._pairing_config = {
            "pairing_anchor": pairing_anchor,
            "source_selection": source_selection,
            "target_selection": target_selection,
            "identity_condition": identity_condition,
            "pair_schedule": pair_schedule,
            "target_sampling": target_sampling,
            "seed": pairing_seed,
            "control_value": control_value,
            "perturbation_mode": next_cell_pred == "pert" and not prediction_only,
            "ot_params": ot_params if use_ot and next_cell_pred == "pert" and not prediction_only else None,
        }
        self._set_pairing(indices, source_indices, target_indices)
        self.expr_layer = expr_layer
        
        # Mappings
        self.cell_type_to_index = cell_type_to_index if cell_type_to_index is not None else {t: i for i, t in enumerate(self.adata.obs['celltype'].unique())}
        self.genotype_to_index = genotype_to_index if genotype_to_index is not None else {t: i for i, t in enumerate(self.adata.obs['genotype'].unique())}
        self.ps_columns = ps_columns or [] 
        self.ps_columns = list(self.ps_columns) # must be a list 
        self.sf = _get_sf(self.adata.layers[self.expr_layer]) if size_factor_col is None else adata.obs[size_factor_col].values.reshape(-1,1)
        # Kept in the signature for existing callers; identity pairs now always use the source row itself.
        self.only_sample_wt_pert = only_sample_wt_pert
        if self.next_cell_pred == "lochness" and not self.prediction_only:
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

    def _set_pairing(self, indices, source_indices=None, target_indices=None):
        self.pairing = ContextAwarePairing(
            self.adata,
            indices,
            source_indices=source_indices,
            target_indices=target_indices,
            **self._pairing_config,
        )
        self.indices = self.pairing.anchor_indices

    @property
    def pairs(self):
        return self.pairing.pairs

    def set_new_indices(self, indices):
        self._set_pairing(np.asarray(indices, dtype=np.int64))

    def get_adata_subset(self, next_cell_pred = 'identity'):
        assert next_cell_pred in ['pert', 'identity', "lochness"], 'next_cell_pred can only be identity or pert or lochness'

        if next_cell_pred == "identity" :
            return self.adata[self.indices,].copy()
        elif next_cell_pred == "lochness":
            adata_small = self.adata[self.indices,].copy()
            adata_small.obs['genotype_next'] = adata_small.obs['genotype']
            return adata_small
        else:
            pairs = self.pairing.pairs or tuple(self.pairing.resolve(i) for i in range(len(self.pairing)))
            source_indices = [pair.source_idx for pair in pairs]
            target_indices = [pair.target_idx for pair in pairs]
            adata_small = self.adata[source_indices,].copy()
            adata_small.obs['genotype_next'] = [pair.condition_label for pair in pairs]
            adata_small.obs['next_cell_id'] = self.adata.obs.index[target_indices]
            adata_small.layers['next_expr'] = self.adata.layers[self.expr_layer][target_indices]
            return adata_small
    

    def __len__(self):
        return len(self.pairing)
    
    def __getitem__(self, idx: int):
        """
        Retrieves one sample from the dataset. This is where on-the-fly processing happens.
        """

        anchor_idx = self.indices[idx]
        pair = None
        if self.prediction_only:
            current_cell_global_idx = int(anchor_idx)
        else:
            pair = self.pairing.resolve(idx)
            current_cell_global_idx = pair.source_idx

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

        if self.prediction_only:
            sample = {
                "expr": current_expr,
                "genes": curr_gene,
                "celltype_labels": self.cell_type_to_index.get(current_cell_celltype, 0),
                "perturbation_labels": self.genotype_to_index.get(current_cell_genotype, 0),
                "batch_labels": current_cell_batch_label,
                "sf": self.sf[current_cell_global_idx],
                "index": current_cell_global_idx,
                "name": current_cell_idx,
            }
            if 'genotype_next' in self.adata.obs.columns:
                next_pert = self.adata.obs.at[current_cell_idx, 'genotype_next']
                sample["perturbation_labels_next"] = self.genotype_to_index.get(next_pert, 0)
            return sample

        # 3. Resolve the target selected by the pairing policy
        assert pair is not None
        next_cell_global_idx = pair.target_idx
        next_cell_id = self.adata.obs.index[next_cell_global_idx]
        next_pert_label_str = pair.condition_label
        
        # 4. Get expression data for the next cell
        next_expr = self.adata.layers[binned_layer_key][next_cell_global_idx]

        next_gene = self.adata.var.index

        if issparse(next_expr):
            next_expr = next_expr.toarray().flatten()
        current_sf = self.sf[current_cell_global_idx]
        next_sf = self.sf[next_cell_global_idx]
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
            "sf": current_sf,
            "sf_next": next_sf,
            "index": current_cell_global_idx,
            "next_index": next_cell_global_idx,
            'name': current_cell_idx,
            'next_name': next_cell_id
        }


class PertBatchCollator:
    """
    A collate function for the DataLoader that tokenizes, pads, and masks batches on the fly.
    """
    def __init__(self, vocab: object, gene_ids: np.ndarray, full_tokenize: bool = False, hvg_inds = None, **config):
        self.config = config
        self.vocab = vocab
        self.gene_ids = gene_ids # vector of gene ids
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
        self.nonzero_prop = config.get('nonzero_prop', 0.7)
        self.sampling_mode = config.get('sampling_mode', 'simple')
        self.fix_nonzero_prop = config.get('fix_nonzero_prop', False)
        self.non_hvg_size = min(config.get('non_hvg_size', 1000), len(hvg_inds[1])) if hvg_inds is not None else 0
        self.hvg_inds = hvg_inds
        self.deterministic = config.get('deterministic', False)
        self.prediction_only = config.get('prediction_only', False)
        self.seed = config.get('seed', 0)

    def __call__(self, batch: list) -> dict:
        """
        Processes a list of samples from the Dataset into a single batch tensor.
        """

        expr_list = [item['expr'] for item in batch]
        if not self.prediction_only:
            expr_next_list = [item['expr_next'] for item in batch]

        # 2. Tokenize and pad the expression data for the current batch
        # max seq len determines the context window for pertTF transformer modeling
        # during validation and predictions, this window may be around all genes with expression
        max_seq_len = self.max_seq_len if not self.full_tokenize else len(self.gene_ids) + self.append_cls

        # TODO: These functions may need to be modified to accomodate inputs w differing number of genes in the future
        expr_mat = np.array(expr_list)
        rng = np.random.default_rng(self.seed) if self.deterministic else None
        tokenized, gene_idx_list = tokenize_and_pad_batch(
            expr_mat, self.gene_ids, max_len=max_seq_len,cls_token=self.cls_token,
            vocab=self.vocab, pad_token=self.pad_token, pad_value=self.pad_value,
            append_cls=self.append_cls, include_zero_gene=self.include_zero_gene, 
            cls_value=self.cls_value, sampling_mode = self.sampling_mode,
            fix_nonzero_prop=self.fix_nonzero_prop, nonzero_prop=self.nonzero_prop,
            hvg_inds = self.hvg_inds, non_hvg_size= self.non_hvg_size,
            rng=rng,
        )
        if not self.prediction_only:
            expr_mat_next = np.array(expr_next_list)
            tokenized_next, _ = tokenize_and_pad_batch(
                expr_mat_next, self.gene_ids, max_len=max_seq_len, cls_token=self.cls_token,
                vocab=self.vocab, pad_token=self.pad_token, pad_value=self.pad_value,
                append_cls=self.append_cls, include_zero_gene=self.include_zero_gene,
                sample_indices=gene_idx_list,
                cls_value=self.cls_value, sampling_mode = self.sampling_mode,
                fix_nonzero_prop=self.fix_nonzero_prop, nonzero_prop=self.nonzero_prop,
                hvg_inds = self.hvg_inds, non_hvg_size= self.non_hvg_size
            )
        
        # 3. Apply random masking for this batch
        masked_values = random_mask_value(
            tokenized["values"], mask_ratio=0 if self.deterministic or self.prediction_only else self.mask_ratio,
            mask_value=self.mask_value, pad_value=self.pad_value,
            cls_value= self.cls_value, rng=rng,
        )

        # 4. Collate all other labels into tensors
        cls_vec = np.array([self.cls_value for i in range(len(batch))]).reshape(-1,1)
        full_gene_id = np.insert(self.gene_ids, 0, self.vocab[self.cls_token]) if self.append_cls else self.gene_ids
        full_gene_id = torch.from_numpy(full_gene_id).long()
        expr_mat = expr_mat if not self.append_cls else np.hstack([cls_vec, expr_mat])

        collated_batch = {
            "gene_ids": tokenized["genes"],
            "values": masked_values,
            "target_values": tokenized["values"],
            "full_expr": torch.Tensor(expr_mat),
            "full_gene_ids": torch.stack([full_gene_id for i in range(len(batch))], dim = 0)
        }
        if not self.prediction_only:
            expr_mat_next = expr_mat_next if not self.append_cls else np.hstack([cls_vec, expr_mat_next])
            collated_batch.update({
                "next_gene_ids": tokenized_next["genes"],
                "target_values_next": tokenized_next["values"],
                "full_expr_next": torch.Tensor(expr_mat_next),
            })
        
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

        self.hvg_inds = None
        n_hvg = config.get('n_hvg', 3000)
        if config.get('sampling_mode', 'simple') == 'hvg':
            self.hvg_col = config.get('hvg_col', 'highly_variable')
            assert self.hvg_col in adata.var.keys(), 'adata must have calculated HVGs or adata.var must have hvg_col'
            n_hvg = min(self.adata.var[self.hvg_col].sum(), n_hvg)
            non_hvg = min(len(self.gene_ids) - n_hvg, config.get('non_hvg_size', 1000))
            self.config.update({'max_seq_len': n_hvg + non_hvg + config.get('append_cls', True)}, allow_val_change=True)
            print(f'sampling_mode is hvg, sampling {n_hvg} HVGs + {non_hvg} non-HVGs for training')
            self.hvg_inds = (np.where(self.adata.var[self.hvg_col])[0], np.where(~self.adata.var[self.hvg_col])[0])
        
        add_batch_info(self.adata)
        self.num_batch_types = len(self.adata.obs["batch_id"].unique())
        # The collators can be created once and reused
        ## first collator is the training collator, with a context window set in config
        self.collator = PertBatchCollator(self.vocab, self.gene_ids, hvg_inds = self.hvg_inds, **config)
        ## full collator may be used for validation or inference 
        ## This may be very slow for full gene set, scaling is roughly 2x context length -> 3.6x time, 3-4x more memory
        self.full_token_collator = PertBatchCollator( self.vocab, self.gene_ids, full_tokenize=True, hvg_inds = self.hvg_inds, **config)
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

    def _create_dataset_from_indices(self, indices, **pairing_config):
        """A helper function to create PertTFDataset from underlying adata."""
        perttf_dataset = PertTFDataset(
            self.adata, 
            indices=indices, 
            # ot parameters
            use_ot=self.config.use_ot, 
            ot_params= self.config.get('ot_params', {}),
            # other parameters
            cell_type_to_index=self.cell_type_to_index, 
            genotype_to_index=self.genotype_to_index,
            ps_columns=self.ps_columns, 
            ps_columns_perturbed_genes = self.ps_columns_perturbed_genes, 
            next_cell_pred=self.next_cell_pred_type ,  
            additional_ps_dict = self.additional_ps_dict,  
            expr_layer=self.expr_layer, 
            only_sample_wt_pert=self.only_sample_wt_pert,
            size_factor_col = self.config.get('size_factor_col', None),
            **pairing_config,
        )
        return perttf_dataset

    def _create_loaders_from_dataset(self, dataset, full_token_collator = False, shuffle=True, deterministic=False, seed=0):
        """A helper function to create dataloaders from PertTFDataset."""    
        if deterministic:
            collator_config = dict(self.config)
            collator_config.update({'deterministic': True, 'seed': seed})
            collator = PertBatchCollator(
                self.vocab,
                self.gene_ids,
                hvg_inds=self.hvg_inds,
                **collator_config,
            )
        else:
            collator = self.collator if not full_token_collator else self.full_token_collator
        loader = DataLoader(
            dataset, batch_size=self.config.batch_size, shuffle=shuffle,
            num_workers=8, collate_fn=collator, pin_memory=True
        )
        return loader

    def get_data_w_loader(
        self,
        indices=None,
        full_data=False,
        full_token=False,
        pairing_config=None,
        shuffle=True,
        deterministic=False,
        seed=0,
    ):
        indices = self.indices if indices is None and full_data else indices
        data = self._create_dataset_from_indices(
            indices,
            **(pairing_config or {}),
        )
        loader = self._create_loaders_from_dataset(
            data,
            full_token_collator=full_token,
            shuffle=shuffle,
            deterministic=deterministic,
            seed=seed,
        )
        return data, loader

    def get_train_valid_loaders(self, test_size: float = 0.1, train_indices = None, valid_indices = None, full_token_validate  = False, random_state = None):
        """Provides a single, standard train/validation split."""
        print(f"Creating a single train/validation split (test_size={test_size})...")
        if train_indices is None or valid_indices is None:
            indices = np.arange(self.adata.n_obs)
            train_indices, valid_indices = train_test_split(indices, test_size=test_size, shuffle=True, random_state=random_state)
        else:
            if len(set(train_indices).intersection(valid_indices)) > 0:
                print('WARNING: training data and validation data are not separate, this may be okay for perturbation if the shared samples are ctrls')
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
