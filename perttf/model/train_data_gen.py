from typing import List, Tuple, Dict, Union, Optional, Literal
import random
import torch
import numpy as np
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader

from scipy.sparse import issparse

from anndata import AnnData
from sklearn.model_selection import train_test_split

from torchtext.vocab import Vocab
from torchtext._torchtext import (
    Vocab as VocabPybind,
)

import scgpt as scg
from utils.custom_tokenizer import tokenize_and_pad_batch, random_mask_value
from scgpt import SubsetsBatchSampler

def add_pred_layer(adata: AnnData, 
        binned_layer_key: Optional[str] = 'X_binned', 
        next_layer_key: Optional[str] = 'X_binned_next',
        next_cell_pred: Literal["identity","pert"] = "identity",
        ps_columns: Optional[str] = None) -> Dict:
    """
    format controls the different input value wrapping, including categorical
    binned style, fixed-sum normalized counts, log1p fixed-sum normalized counts, etc.

    Args:

    adata (:class:`AnnData`):
        The :class:`AnnData` object to preprocess.
    binned_layer_key (:class:`str`, optional):
        The key of :class:`AnnData.obs` to use for expression layer. Default is the binned expression layer.
    next_layer_key (:class:`str`, optional):
        The key of :class:`AnnData.obs` to use for next-stage expression layer. Default is the binned expression layer.
    copy_data (:class:`bool`, optional):
        Whether to directly copy the data

    Returns:
    (current_counts, next_counts, celltypes_labels_0, perturbation_labels_0, batch_ids_0, celltypes_labels_next, perturbation_labels_next, adata)
    """
    if next_cell_pred == "identity":
        adata_p = adata
    else:
        #adata_p = adata[adata.obs['genotype']=='WT']
        adata_p = adata

    all_counts_0 = (adata_p.layers[binned_layer_key].toarray() if issparse(adata_p.layers[binned_layer_key]) else adata_p.layers[binned_layer_key])

    if "celltype" in adata_p.obs.columns:
        celltypes_labels_0 = adata_p.obs["celltype"].tolist()  # make sure count from 0
    else:
        celltypes_labels_0 = random.choices( [0,1], k=adata_p.shape[0])
    #num_types = len(set(celltypes_labels))
    celltypes_labels_0 = np.array(celltypes_labels_0)

    if "genotype" in adata_p.obs.columns:
        perturbation_labels_0 = adata_p.obs["genotype"].tolist()  # make sure count from 0
    else:
        perturbation_labels_0 = random.choices( [0,1], k=adata_p.shape[0])
    perturbation_labels_0 = np.array(perturbation_labels_0)

    if "batch_id" in adata_p.obs.columns: # and config.DSBN:
        batch_ids_0 = adata_p.obs["batch_id"].tolist()
    else:
        batch_ids_0=random.choices( [0,1], k=adata_p.shape[0])
    batch_ids_0 = np.array(batch_ids_0)
    # right now, just a duplicate of input layers
    # 
    celltypes_labels_next = celltypes_labels_0

    if ps_columns is not None:
        ps_exist_c=[x for x in ps_columns if x in adata_p.obs.columns]
        print('total number of columns used for PS modeling:'+str(len(ps_exist_c)))
        ps_matrix=adata_p.obs[ps_exist_c].values
        ps_matrix=np.nan_to_num(ps_matrix,nan=0.0)
    else:
        ps_exist_c=["PS"]
        ps_matrix= [0.0] * adata_p.shape[0]# a shape of 0
        ps_matrix = np.array(ps_matrix) # must be an np array
    #
    if next_cell_pred == "identity":
        perturbation_labels_next = perturbation_labels_0
        input_layers=adata_p.layers[binned_layer_key] #.copy()
        retdict={
            'expmat': all_counts_0,
            'expmat_next': input_layers,
            'celllabel': celltypes_labels_0,
            'pertlabel': perturbation_labels_0, 
            'batchid': batch_ids_0, 
            'celllabel_next': celltypes_labels_next, 
            'pertlabel_next': perturbation_labels_next, 
            'adata': adata_p,
            'ps': ps_matrix,
            'ps_next': ps_matrix,
            'ps_names': ps_exist_c,
        }
        return retdict
        #return (all_counts_0,input_layers, celltypes_labels_0, perturbation_labels_0, batch_ids_0, celltypes_labels_next, perturbation_labels_next, adata_p)
    
    # predict the next cell type
    obsf=adata.obs
    next_cell_dict = {}
    genotype_pool = list(set(obsf['genotype']))
    for candidate_celltype in set(obsf['celltype']):
        next_cell_dict[candidate_celltype] = {}
        for candidate_genotype in set(obsf['genotype']):
            obsf_sel = obsf[(obsf['celltype']==candidate_celltype) & (obsf['genotype']==candidate_genotype)]
            included_cells = obsf_sel.index.to_list()
            if len(included_cells) > 0:
                next_cell_dict[candidate_celltype][candidate_genotype] = included_cells

    # adata_p = adata[adata.obs['genotype']=='WT']

    target_pert=[]
    target_cell_id=[]
    for this_cell in adata_p.obs.index.values:
        this_cell_type = adata_p.obs.loc[this_cell]['celltype']
        this_geno_type = adata_p.obs.loc[this_cell]['genotype']
        if this_geno_type == 'WT': # randomly select a different genotype
            next_pert_value=random.choice(list(next_cell_dict[this_cell_type].keys()))
        else: # choose the same perturb as the next pert (i.e., no combination perturbation) for training
            next_pert_value = this_geno_type
        target_pert.append(next_pert_value)
        next_cell = random.choice(next_cell_dict[this_cell_type][next_pert_value])
        target_cell_id.append(next_cell)

    adata_p.obs['genotype_next'] = target_pert # random select the next perturbations
    adata_p.obs['next_cell_id'] = target_cell_id
    
    #(adata_p.layers[binned_layer_key].toarray() if issparse(adata_p.layers[binned_layer_key]) else adata_p.layers[binned_layer_key])

    input_layers=all_counts_0 # adata.layers[binned_layer_key] #.copy()
    target_cell_id_index=obsf.index.get_indexer(target_cell_id)
    target_layers=input_layers[target_cell_id_index]
    perturbation_labels_next = target_pert
    perturbation_labels_next = np.array(perturbation_labels_next)
    # return the ps matrix next
    ps_matrix_next = ps_matrix[target_cell_id_index]
    # instead of returning a tuplex, return a dictionary 
    retdict={
        'expmat': all_counts_0,
        'expmat_next': target_layers,
        'celllabel': celltypes_labels_0,
        'pertlabel': perturbation_labels_0, 
        'batchid': batch_ids_0, 
        'celllabel_next': celltypes_labels_next, 
        'pertlabel_next': perturbation_labels_next, 
        'adata': adata_p,
        'ps':ps_matrix,
        'ps_next': ps_matrix_next,
        'ps_names': ps_exist_c,
    }
    return retdict
    #return (all_counts_0,target_layers, celltypes_labels_0, perturbation_labels_0, batch_ids_0, celltypes_labels_next, perturbation_labels_next, adata_p)


def add_pred_layer_old(adata: AnnData, binned_layer_key: Optional[str] = 'X_binned', next_layer_key: Optional[str] = 'X_binned_next') -> Dict:
    """
    format controls the different input value wrapping, including categorical
    binned style, fixed-sum normalized counts, log1p fixed-sum normalized counts, etc.

    Args:

    adata (:class:`AnnData`):
        The :class:`AnnData` object to preprocess.
    binned_layer_key (:class:`str`, optional):
        The key of :class:`AnnData.obs` to use for expression layer. Default is the binned expression layer.
    next_layer_key (:class:`str`, optional):
        The key of :class:`AnnData.obs` to use for next-stage expression layer. Default is the binned expression layer.
    """

    # right now, just a duplicate of input layers
    input_layers=adata.layers[binned_layer_key].copy()
    return input_layers,adata

    n_bin_pseudotome=5 # split pseudotime bins

    obsf=adata.obs
    import pandas as pd
    obsf['pseudotime_bin'] = pd.cut(obsf['palantir_pseudotime'], bins=np.linspace(0, 1, num=n_bin_pseudotome), labels=False) # split [0,1] into n bins and calculate bin number for palantir_pseudotime column

    p_prob=adata.obsm['palantir_fate_probabilities']
    n_diff_cell_types=p_prob.shape[1]
    p_prob['max_index'] = np.argmax(p_prob.values, axis=1) # find the index with the largest value for each row

    obsf_2=obsf[['palantir_pseudotime','palantir_entropy','pseudotime_bin']].join(p_prob)

    target_cell_id=[]
    is_valid_next=[]
    n_no_next = 0
    for this_cell in obsf_2.index.values:
        t_ps_bin=obsf_2['pseudotime_bin'].loc[this_cell]
        t_max_index=obsf_2['max_index'].loc[this_cell]
        t_entrophy=obsf_2['palantir_entropy'].loc[this_cell]
        cell_pool=[]
        if t_ps_bin == 0: # pseudo-time =0; can be any cell in the next bin
            # cell_pool=obsf_2.index.values[obsf_2['pseudotime_bin'] == t_ps_bin+1 & (obsf_2['palantir_entropy'] < t_entrophy) ]
            cell_pool=obsf_2.index.values[obsf_2['pseudotime_bin'] == t_ps_bin+1  ]  # do not use entrophy
        elif t_ps_bin == n_bin_pseudotome - 2: # pseudo-time =last bin; can be any cell in the current bin with the same fate
            #cell_pool=obsf_2.index.values[ (obsf_2['pseudotime_bin'] == t_ps_bin) & (obsf_2['max_index']==t_max_index) & (obsf_2['palantir_entropy'] < t_entrophy)]
            cell_pool=obsf_2.index.values[ (obsf_2['pseudotime_bin'] == t_ps_bin) & (obsf_2['max_index']==t_max_index) ]
        else:
            #cell_pool=obsf_2.index.values[ (obsf_2['pseudotime_bin'] == t_ps_bin+1) & (obsf_2['max_index']==t_max_index) & (obsf_2['palantir_entropy'] < t_entrophy)]
            cell_pool=obsf_2.index.values[ (obsf_2['pseudotime_bin'] == t_ps_bin+1) & (obsf_2['max_index']==t_max_index)]
        if len(cell_pool) > 0:
            target_cell_id.append(random.choice(cell_pool))
            is_valid_next.append(True)
        else:
            target_cell_id.append(this_cell)
            n_no_next = n_no_next + 1
            is_valid_next.append(False)

    adata.obs['next_cell_id'] = target_cell_id
    adata.obs['is_valid_next'] = is_valid_next

    input_layers=adata.layers[binned_layer_key].copy()
    target_cell_id_index=obsf_2.index.get_indexer(target_cell_id)
    target_layers=input_layers[target_cell_id_index]

    #adata.layers[next_layer_key]=target_layers


    return target_layers





def produce_training_datasets(adata_input, config,
                              input_layer_key = "X_binned",
                              next_layer_key = "X_binned_next",
                              next_cell_pred: Literal["identity","pert"] = "identity",
                              cell_type_to_index = None,
                              genotype_to_index = None,
                              vocab = None,
                              ps_columns = None,
                              logger = scg.logger):
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
                                  
    # add necessary columns to adata
    adata_input.var["gene_name"] = adata_input.var.index.tolist()

    # set up these random values such that they don't get errors later on during training
    if "batch" not in adata_input.obs.columns: 
        batch_ids_0=random.choices( [0,1], k=adata_input.shape[0])
        adata_input.obs["batch"]=batch_ids_0
    adata_input.obs["str_batch"] = adata_input.obs["batch"]
    adata_input.obs["str_batch"] = adata_input.obs["str_batch"].astype(str)
    adata_input.obs["batch_id"] = adata_input.obs["str_batch"].astype("category").cat.codes.values
    #adata.obs["celltype"] = random.choices( [0,1], k=adata.shape[0])
    #adata.obs["celltype"] = adata.obs["celltype"].astype("category")

    # further expand the next layer prediction
    n_rounds = config.n_rounds if hasattr(config,"n_rounds") else 1
    adata = None
    genes = None
    next_counts = None
    all_counts = None
    celltypes_labels = None 
    perturbation_labels = None
    batch_ids = None
    num_batch_types = -1
    celltypes_indexes = None
    perturbation_indexes = None
    celltypes_indexes_next = None
    perturbation_indexes_next = None
    batch_indexes = None
    cell_ids = None
    index_rounds=None
    #
    ps_scores=None
    ps_scores_next=None

    if cell_type_to_index is None:
        cell_type_to_index = {cell_type: index for index, cell_type in enumerate(set(adata_input.obs["celltype"].tolist()))}

    if genotype_to_index is None:
        genotype_to_index = {genotype: index for index, genotype in enumerate(set(adata_input.obs["genotype"].tolist()))}
    else:
        adata_input = adata_input[adata_input.obs['genotype'].isin(genotype_to_index.keys())]
        
    n_cls = len(cell_type_to_index)
    n_perturb = len(genotype_to_index)

    for ni in range(n_rounds):
        # predict the next state of a cell
        #next_counts_0,adata_0 = add_pred_layer(adata_input,next_cell_pred=next_cell_pred)
        print(f'rounds: {ni}')
        #(all_counts_0,next_counts_0, celltypes_labels_0, perturbation_labels_0, batch_ids_0, celltypes_labels_next, perturbation_labels_next,  adata_0) = add_pred_layer(adata_input,next_cell_pred=next_cell_pred)
        add_l_ret = add_pred_layer(adata_input,next_cell_pred=next_cell_pred, ps_columns = ps_columns)
        
        all_counts_0 = add_l_ret['expmat']
        next_counts_0 = add_l_ret['expmat_next'] 
        celltypes_labels_0 = add_l_ret['celllabel'] 
        perturbation_labels_0 = add_l_ret['pertlabel'] 
        batch_ids_0 = add_l_ret['batchid'] 
        celltypes_labels_next = add_l_ret['celllabel_next'] 
        perturbation_labels_next = add_l_ret['pertlabel_next']  
        adata_0 = add_l_ret['adata']
        ps_0 = add_l_ret['ps']
        ps_next_0 = add_l_ret['ps_next']
        ps_names = add_l_ret['ps_names']
        
        print('adding next layers...')
        #all_counts_0 = (adata_0.layers[input_layer_key].A if issparse(adata_0.layers[input_layer_key]) else adata_0.layers[input_layer_key])

        # generate cell type indexs and perturbation indexes


        celltypes_indexes_0 = [cell_type_to_index[cell_type] for cell_type in celltypes_labels_0]
        celltypes_indexes_0 = np.array(celltypes_indexes_0)
        # Now celltypes_indexes contains the numerical representation of cell types
        #print(celltypes_indexes[:10]) # Example: print first 10 indexes

        perturbation_indexes_0 = [genotype_to_index[genotype] for genotype in perturbation_labels_0]
        perturbation_indexes_0 = np.array(perturbation_indexes_0)

        # add variables for next prediction
        celltypes_indexes_next_0 = [cell_type_to_index[cell_type] for cell_type in celltypes_labels_next]
        celltypes_indexes_next_0 = np.array(celltypes_indexes_next_0)

        perturbation_indexes_next_0 = [genotype_to_index[genotype] for genotype in perturbation_labels_next]
        perturbation_indexes_next_0 = np.array(perturbation_indexes_next_0)

        ps_0 = np.array(ps_0)
        ps_next_0 = np.array(ps_next_0)
        # TODO: need to make sure matrices are either not scipy sparse or implement sparse concatenations below
        if adata is None:
            adata = adata_0 #.copy()
            next_counts = next_counts_0
            all_counts = all_counts_0
            adata.layers[next_layer_key]=next_counts
            genes = adata.var["gene_name"].tolist()
            celltypes_labels = celltypes_labels_0
            perturbation_labels = perturbation_labels_0
            batch_ids = batch_ids_0
            num_batch_types = len(set(batch_ids))
            celltypes_indexes = celltypes_indexes_0
            perturbation_indexes = perturbation_indexes_0
            celltypes_indexes_next = celltypes_indexes_next_0
            perturbation_indexes_next = perturbation_indexes_next_0
            cell_ids = adata.obs.index.values
            index_rounds = np.array([ni]*len(celltypes_labels_0))
            ps_scores = ps_0
            ps_scores_next = ps_next_0
        else:
            adata_0.obs.index = adata_0.obs.index + "-r"+str(ni)
            adata = adata.concatenate(adata_0,batch_key='batch_merged_rounds',index_unique=None)
            
            # create a breakpoint here
            #import pdb; pdb.set_trace()
            next_counts = np.concatenate([next_counts, next_counts_0], axis=0)
            all_counts = np.concatenate([all_counts, all_counts_0], axis=0)
            celltypes_labels = np.concatenate([celltypes_labels, celltypes_labels_0], axis=0)
            perturbation_labels = np.concatenate([perturbation_labels, perturbation_labels_0], axis=0)
            batch_ids = np.concatenate([batch_ids, batch_ids_0], axis=0)
            celltypes_indexes = np.concatenate([celltypes_indexes, celltypes_indexes_0], axis=0)
            perturbation_indexes = np.concatenate([perturbation_indexes, perturbation_indexes_0], axis=0)
            celltypes_indexes_next = np.concatenate([celltypes_indexes_next, celltypes_indexes_next_0], axis=0)
            perturbation_indexes_next = np.concatenate([perturbation_indexes_next, perturbation_indexes_next_0], axis=0)
            cell_ids = np.concatenate([cell_ids, adata_0.obs.index.values],axis=0)
            index_rounds = np.concatenate([index_rounds, np.array([ni]*len(celltypes_labels_0))], axis=0)
            #.obs['batch_merged_rounds'] = index_rounds
            ps_scores = np.concatenate([ps_scores, ps_0], axis=0)
            ps_scores_next = np.concatenate([ps_scores_next, ps_next_0], axis=0)

        #cell_ids = adata.obs.index.values
            
            # add next layers

        # or just duplicate the data
        #next_counts = all_counts.copy()

        #next_counts = (adata.layers[next_layer_key].A if issparse(adata.layers[next_layer_key]) else adata.layers[next_layer_key])

        #print(perturbation_indexes[:10])

    #n_cls = len(set(celltypes_labels)) if config.cell_type_classifier else 1
    #n_cls
    #n_perturb = len(set(perturbation_labels)) if config.perturbation_classifier_weight > 0 else 1
    print('splitting train and test data...')
    # now, split train and test data
    (
        train_data, valid_data, # all_counts 
        train_celltype_labels, valid_celltype_labels, # celltypes_indexes
        train_perturbation_labels, valid_perturbation_labels,
        train_batch_labels,  valid_batch_labels,
        train_data_next, valid_data_next,
        train_celltype_labels_next, valid_celltype_labels_next, # celltypes_indexes_next
        train_perturbation_labels_next, valid_perturbation_labels_next, # perturbation_indexes_next
        cell_ids_train, cell_ids_valid,
        ps_train, ps_valid,
        ps_next_train, ps_next_valid,
    ) = train_test_split(
        all_counts, celltypes_indexes, perturbation_indexes, batch_ids, next_counts, 
        celltypes_indexes_next, perturbation_indexes_next,
        cell_ids, ps_scores, ps_scores_next,
        test_size=0.1, shuffle=True
    )

    adata.obs['is_in_training']=adata.obs.index.isin(cell_ids_train)
    #adata_small=adata[adata.obs['is_in_training']==True] # only consider training data
    adata_small=adata[adata.obs['is_in_training']==False] # only consider validation data; note that for pert, this only contains adata of the round 0
    
    #for ni in range(n_rounds):
    #    print(f"round {ni}")
    #    adata_input_slice=adata_input[adata_input.obs.index.isin(adata_small.obs.index.values)]
    #    
    #    round_next_c,adata_rets = add_pred_layer(adata_input_slice,next_cell_pred=next_cell_pred)
    #
    #    round_all_c = adata_small.layers[input_layer_key].copy()
    #
    #    train_data = np.concatenate([train_data, round_all_c], axis=0)
    #    train_data_next = np.concatenate([train_data_next, round_next_c], axis=0)

    #train_celltype_labels = np.concatenate([train_celltype_labels] * (n_rounds+1))
    #train_perturbation_labels = np.concatenate([train_perturbation_labels] * (n_rounds+1))
    #train_batch_labels = np.concatenate([train_batch_labels] * (n_rounds+1))
    #cell_ids_train = np.concatenate([cell_ids_train] * (n_rounds + 1))


    # prompt: shuffle the rows of all_counts and next_counts with the same order

    # Generate a random permutation of indices
    indices = np.arange(train_data.shape[0])
    np.random.shuffle(indices)

    # Shuffle both arrays using the same indices
    train_data = train_data[indices]
    train_data_next = train_data_next[indices]
    train_celltype_labels = train_celltype_labels[indices]
    train_perturbation_labels = train_perturbation_labels[indices]
    train_batch_labels = train_batch_labels[indices]
    train_celltype_labels_next = train_celltype_labels_next[indices]
    train_perturbation_labels_next = train_perturbation_labels_next[indices]
    cell_ids_train = cell_ids_train[indices]
    ps_train = ps_train[indices]
    ps_next_train = ps_next_train[indices]

    if config.per_seq_batch_sample  and "batch_id" in adata.obs.columns: # and config.DSBN
        # sort the adata by batch_id in advance
        #adata_sorted = adata[adata.obs["batch_id"].argsort()].copy()
        #adata_sorted.obs
        adata_sorted = adata_small[adata_small.obs["batch_id"].argsort()].copy()
    else:
        #adata_sorted = adata.copy()
        adata_sorted = adata_small.copy()

    # construct vocab
    if vocab is None:
        vocab = Vocab(
                VocabPybind(genes + config.special_tokens, None)
        )  # bidirectional lookup [gene <-> int]
        vocab.set_default_index(vocab["<pad>"])
    gene_ids = np.array(vocab(genes), dtype=int)

    # construct tokenized data
    print('tokenize data...')
    tokenized_train, gene_idx_list_train= tokenize_and_pad_batch(
        train_data,
        gene_ids,
        max_len=config.max_seq_len,
        vocab=vocab,
        pad_token=config.pad_token,
        pad_value=config.pad_value,
        append_cls=True,  # append <cls> token at the beginning
        include_zero_gene=True,
    )
    tokenized_valid, gene_idx_list_valid = tokenize_and_pad_batch(
        valid_data,
        gene_ids,
        max_len=config.max_seq_len,
        vocab=vocab,
        pad_token=config.pad_token,
        pad_value=config.pad_value,
        append_cls=True,
        include_zero_gene=True,
    )

    tokenized_train_next, _ = tokenize_and_pad_batch(
        train_data_next,
        gene_ids,
        max_len=config.max_seq_len,
        vocab=vocab,
        pad_token=config.pad_token,
        pad_value=config.pad_value,
        append_cls=True,  # append <cls> token at the beginning
        include_zero_gene=True,
         sample_indices= gene_idx_list_train
    )
    tokenized_valid_next, _ = tokenize_and_pad_batch(
        valid_data_next,
        gene_ids,
        max_len=config.max_seq_len,
        vocab=vocab,
        pad_token=config.pad_token,
        pad_value=config.pad_value,
        append_cls=True,
        include_zero_gene=True, 
        sample_indices= gene_idx_list_valid

    )
    if logger is not None:
        logger.info(
            f"train set number of samples: {tokenized_train['genes'].shape[0]}, "
            f"\n\t feature length: {tokenized_train['genes'].shape[1]}"
        )
        logger.info(
            f"valid set number of samples: {tokenized_valid['genes'].shape[0]}, "
            f"\n\t feature length: {tokenized_valid['genes'].shape[1]}"
        )


    # construct return value
    ret_dict={
        'adata': adata, # original adata
        'adata_sorted': adata_sorted, # training adata
        'vocab': vocab,
        'n_perturb': n_perturb,
        'n_cls': n_cls,
        'genes': genes,
        'gene_ids': gene_ids,
        'num_batch_types': num_batch_types,
        'cell_type_to_index': cell_type_to_index,
        'genotype_to_index': genotype_to_index,
        'train_data': train_data,
        'valid_data': valid_data,
        'train_data_next': train_data_next,
        'valid_data_next': valid_data_next,
        'train_celltype_labels': train_celltype_labels,
        'valid_celltype_labels': valid_celltype_labels,
        'train_perturbation_labels': train_perturbation_labels,
        'valid_perturbation_labels': valid_perturbation_labels,
        'train_batch_labels': train_batch_labels,
        'valid_batch_labels': valid_batch_labels,
        'train_celltype_labels_next': train_celltype_labels_next, # next cell type
        'valid_celltype_labels_next': valid_celltype_labels_next, 
        'train_perturbation_labels_next': train_perturbation_labels_next, # next genotype
        'valid_perturbation_labels_next': valid_perturbation_labels_next,
        'cell_ids_train': cell_ids_train,
        'cell_ids_valid': cell_ids_valid,
        'tokenized_train': tokenized_train,
        'tokenized_valid': tokenized_valid,
        'tokenized_train_next': tokenized_train_next,
        'tokenized_valid_next': tokenized_valid_next,
        'ps_train': ps_train,
        'ps_valid': ps_valid,
        'ps_next_train': ps_next_train,
        'ps_next_valid': ps_next_valid,
        'ps_names': ps_names,
    }
    return ret_dict

def prepare_data(
    data_dict,
    config,
    sort_seq_batch=False,
    epoch = 0) -> Tuple[Dict[str, torch.Tensor]]:

    # extract data
    tokenized_train=data_dict['tokenized_train']
    tokenized_valid=data_dict['tokenized_valid']
    tokenized_train_next=data_dict['tokenized_train_next']
    tokenized_valid_next=data_dict['tokenized_valid_next']
    train_batch_labels=data_dict['train_batch_labels']
    valid_batch_labels=data_dict['valid_batch_labels']
    train_celltype_labels=data_dict['train_celltype_labels']
    valid_celltype_labels=data_dict['valid_celltype_labels']
    train_perturbation_labels=data_dict['train_perturbation_labels']
    valid_perturbation_labels=data_dict['valid_perturbation_labels']
    train_celltype_labels_next=data_dict['train_celltype_labels_next']
    valid_celltype_labels_next=data_dict['valid_celltype_labels_next']
    train_perturbation_labels_next=data_dict['train_perturbation_labels_next']
    valid_perturbation_labels_next=data_dict['valid_perturbation_labels_next']
    #
    ps_train=data_dict['ps_train']
    ps_valid=data_dict['ps_valid']
    ps_next_train=data_dict['ps_next_train']
    ps_next_valid=data_dict['ps_next_valid']

    masked_values_train = random_mask_value(
        tokenized_train["values"],
        mask_ratio=config.mask_ratio,
        mask_value=config.mask_value,
        pad_value=config.pad_value,
    )
    masked_values_valid = random_mask_value(
        tokenized_valid["values"],
        mask_ratio=config.mask_ratio,
        mask_value=config.mask_value,
        pad_value=config.pad_value,
    )
    print(
        f"random masking at epoch {epoch:3d}, ratio of masked values in train: ",
        f"{(masked_values_train == config.mask_value).sum() / (masked_values_train - config.pad_value).count_nonzero():.4f}",
    )

    input_gene_ids_train, input_gene_ids_valid = (
        tokenized_train["genes"],
        tokenized_valid["genes"],
    )
    input_values_train, input_values_valid = masked_values_train, masked_values_valid
    target_values_train, target_values_valid = (
        tokenized_train["values"],
        tokenized_valid["values"],
    )
    target_values_train_next, target_values_valid_next = ( # added
        tokenized_train_next["values"],
        tokenized_valid_next["values"],
    )

    tensor_batch_labels_train = torch.from_numpy(train_batch_labels).long()
    tensor_batch_labels_valid = torch.from_numpy(valid_batch_labels).long()


    tensor_celltype_labels_train = torch.from_numpy(train_celltype_labels).long()#added
    tensor_celltype_labels_valid = torch.from_numpy(valid_celltype_labels).long()#added

    tensor_perturbation_labels_train = torch.from_numpy(train_perturbation_labels).long()#added
    tensor_perturbation_labels_valid = torch.from_numpy(valid_perturbation_labels).long()#added

    tensor_celltype_labels_train_next = torch.from_numpy(train_celltype_labels_next).long()#added
    tensor_celltype_labels_valid_next = torch.from_numpy(valid_celltype_labels_next).long()#added

    tensor_perturbation_labels_train_next = torch.from_numpy(train_perturbation_labels_next).long()#added
    tensor_perturbation_labels_valid_next = torch.from_numpy(valid_perturbation_labels_next).long()#added

    tensor_ps_train=torch.from_numpy(ps_train).float()
    tensor_ps_valid=torch.from_numpy(ps_valid).float()

    tensor_ps_train_next=torch.from_numpy(ps_next_train).float() # now, duplicate ps for next
    tensor_ps_valid_next=torch.from_numpy(ps_next_valid).float()

    if sort_seq_batch:
        train_sort_ids = np.argsort(train_batch_labels)
        input_gene_ids_train = input_gene_ids_train[train_sort_ids]
        input_values_train = input_values_train[train_sort_ids]
        target_values_train = target_values_train[train_sort_ids]
        target_values_train_next = target_values_train_next[train_sort_ids] # added
        tensor_batch_labels_train = tensor_batch_labels_train[train_sort_ids]
        tensor_celltype_labels_train = tensor_celltype_labels_train[train_sort_ids]#added
        tensor_perturbation_labels_train = tensor_perturbation_labels_train[train_sort_ids]#added

        tensor_celltype_labels_train_next = tensor_celltype_labels_train_next[train_sort_ids]#added
        tensor_perturbation_labels_train_next = tensor_perturbation_labels_train_next[train_sort_ids]#added
        tensor_ps_train=tensor_ps_train[train_sort_ids] # added
        tensor_ps_train_next=tensor_ps_train_next[train_sort_ids]

        valid_sort_ids = np.argsort(valid_batch_labels)
        input_gene_ids_valid = input_gene_ids_valid[valid_sort_ids]
        input_values_valid = input_values_valid[valid_sort_ids]
        target_values_valid = target_values_valid[valid_sort_ids]
        target_values_valid_next = target_values_valid_next[valid_sort_ids] # added
        tensor_batch_labels_valid = tensor_batch_labels_valid[valid_sort_ids]
        tensor_celltype_labels_valid = tensor_celltype_labels_valid[valid_sort_ids] #added
        tensor_perturbation_labels_valid = tensor_perturbation_labels_valid[valid_sort_ids] #added

        tensor_celltype_labels_valid_next = tensor_celltype_labels_valid_next[valid_sort_ids] #added
        tensor_perturbation_labels_valid_next = tensor_perturbation_labels_valid_next[valid_sort_ids] #added
        tensor_ps_valid=tensor_ps_valid[valid_sort_ids] #added
        tensor_ps_valid_next=tensor_ps_valid_next[valid_sort_ids]

    train_data_pt = {
        "gene_ids": input_gene_ids_train,
        "values": input_values_train,
        "target_values": target_values_train,
        "target_values_next": target_values_train_next, # added
        "batch_labels": tensor_batch_labels_train,
        "celltype_labels": tensor_celltype_labels_train, #added
        "perturbation_labels": tensor_perturbation_labels_train, #added
        "celltype_labels_next": tensor_celltype_labels_train_next, #added
        "perturbation_labels_next": tensor_perturbation_labels_train_next, #added
        "ps": tensor_ps_train,
        "ps_next": tensor_ps_train_next,

    }
    valid_data_pt = {
        "gene_ids": input_gene_ids_valid,
        "values": input_values_valid,
        "target_values": target_values_valid,
        "target_values_next": target_values_valid_next, # added
        "batch_labels": tensor_batch_labels_valid,
        "celltype_labels": tensor_celltype_labels_valid, #added
        "perturbation_labels": tensor_perturbation_labels_valid, #added
        "celltype_labels_next": tensor_celltype_labels_valid_next, #added
        "perturbation_labels_next": tensor_perturbation_labels_valid_next, #added
        "ps": tensor_ps_valid,
        "ps_next": tensor_ps_valid_next,
    }

    return train_data_pt, valid_data_pt


# dataset
class SeqDataset(Dataset):
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.data = data

    def __len__(self):
        return self.data["gene_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}


# data_loader
def prepare_dataloader(
    data_pt: Dict[str, torch.Tensor],
    batch_size: int,
    config,
    shuffle: bool = False,
    intra_domain_shuffle: bool = False,
    drop_last: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    dataset = SeqDataset(data_pt)

    if config.per_seq_batch_sample:
        # find the indices of samples in each seq batch
        subsets = []
        batch_labels_array = data_pt["batch_labels"].numpy()
        for batch_label in np.unique(batch_labels_array):
            batch_indices = np.where(batch_labels_array == batch_label)[0].tolist()
            subsets.append(batch_indices)
        data_loader = DataLoader(
            dataset=dataset,
            batch_sampler=SubsetsBatchSampler(
                subsets,
                batch_size,
                intra_subset_shuffle=intra_domain_shuffle,
                inter_subset_shuffle=shuffle,
                drop_last=drop_last,
            ),
            num_workers=num_workers,
            pin_memory=True,
        )
        return data_loader

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True,
    )
    return data_loader

