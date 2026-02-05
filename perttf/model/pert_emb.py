
import torch
import os
from typing import Literal

import scanpy as sc
import numpy as np
import pandas as pd
from .train_function import eval_testdata

def load_pert_embedding_from_gears(gears_path, adata, 
                                  intersect_type : Literal["common","gears"] = "common"):
    """
    load pretrained perturbation embeddings from GEARS model
    Args:
        gears_path: path to gears model
        adata:    scanpy object
        intersect_type:    choose the way to handle gears perturbed genes and adata.obs['genotype']. default intersect.
    Returns:
    """
    # load two required files
    gears_model_dict = torch.load(os.path.join(gears_path, 'model.pt'), map_location = torch.device('cpu'))
    pklfile=os.path.join(gears_path, 'pert_gene_list.pkl')
    import pickle
    with open(pklfile, 'rb') as f:
        pkl_data = pickle.load(f)
    #len(pkl_data['pert_gene_list'])
    gears_gene_list=pkl_data['pert_gene_list']

    # extract the perturbation column in adata
    a_genotype_list=adata.obs['genotype'].unique()
    # a_genotype_list
    import numpy as np

    if intersect_type == 'common':
        #common_genotype_list = a_genotype_list[a_genotype_list.isin(gears_gene_list)]
        common_genotype_list = np.intersect1d(a_genotype_list, gears_gene_list)
    elif intersect_type == 'gears':
        common_genotype_list = gears_gene_list
    else:
        common_genotype_list = gears_gene_list

    print('common genes between GEARS embeddings and your adata genotypes: ' + ','.join(common_genotype_list))
    #print('adata genotypes not in GEARS embeddings: ' + ','.join(a_genotype_list[~a_genotype_list.isin(gears_gene_list)]))
    print('adata genotypes not in GEARS embeddings: ' + ','.join([x for x in a_genotype_list if x not in gears_gene_list]))
    #print('adata genotypes not in GEARS embeddings: ' + ','.join(list(set(a_genotype_list) - set(gears_gene_list))))


    import numpy as np
    common_genotype_indices = [gears_gene_list.index(genotype) for genotype in common_genotype_list if genotype in gears_gene_list]
    gears_model_subset = gears_model_dict['pert_emb.weight'][common_genotype_indices,:]
    genotype_index_gears = {genotype: index for index, genotype in enumerate(common_genotype_list)}
    #genotype_index_gears

    # add wild-type label
    genotype_index_gears['WT'] = len(genotype_index_gears)

    # Add a row of zeros at the end of weights so it represents "WT"
    gears_model_subset = np.vstack([gears_model_subset, np.zeros(gears_model_subset.shape[1])])
    # subset adata
    adata_subset = adata[adata.obs['genotype'].isin(genotype_index_gears.keys())]
    # construct return dict
    ret_dict = {'pert_embedding':gears_model_subset,
                'adata_subset':adata_subset,
                'genotype_index_gears':genotype_index_gears,}
    return ret_dict



def load_pert_embedding_to_model(o_model, model_weights, requires_grad = True):
    """
    load pretrained perturbation embeddings to model
    Args:
      omodel: model object
      model_weights: perturbation embeddings, must be a tensor or numpy array whose size is the same as o_model.pert_encoder.embedding.weight.shape
    Returns:
      model: model object with perturbation embeddings loaded
    """
    model_weights_tensor = torch.tensor(model_weights, 
                                             dtype=o_model.pert_encoder.embedding.weight.dtype, 
                                             device=o_model.pert_encoder.embedding.weight.device)
    if model_weights_tensor.shape != o_model.pert_encoder.embedding.weight.shape:
        raise ValueError(f"model_weights_tensor.shape {model_weights_tensor.shape} does not equal to o_model.pert_encoder.embedding.weight.shape {o_model.pert_encoder.embedding.weight.shape}")
    o_model.pert_encoder.embedding.weight.data = model_weights_tensor # torch.tensor(gears_model_subset,dtype=torch.double)
    o_model.pert_encoder.embedding.weight.requires_grad = requires_grad
    return o_model


def generate_pert_embeddings(adata_target, adata_wt, candidate_genes,
                             model, gene_ids, cell_type_to_index, genotype_to_index, vocab,
                             config, device,
                             n_expands_per_epoch = 50,
                             n_epoch = 4,
                             wt_pred_next_label = "WT", ):
    """
    Generate perturbation embeddings for target cells, and calculate cosine similarity between all target cells vs wild-type cells

    Args:
      adata_target: AnnData object of the target cells, where the perturbation will be based on (this is the "source" of the perturbation).
      adata_wt: AnnData object of the wildtype cells. This is considered as the "target" of the perturbation 
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
    adata_bwmerge=sc.concat([adata_target]*n_expands_per_epoch + [adata_wt],axis=0)
    cell_emb_data_all = None
    perturb_info_all = None

    cs_matrix_res = np.zeros((adata_wt.shape[0], adata_target.shape[0] * n_expands_per_epoch, n_epoch))
    # loop over epochs
    for n_round in range(n_epoch):
        # assign genoytpe_next
        gt_next_1 = np.random.choice(list(candidate_genes), size = adata_target.shape[0] * n_expands_per_epoch)
        #adata_test_gw_wtmerge.obs.loc[adata_test_gw_wtmerge.obs['genotype']=='WT' ,'genotype_next'].value_counts()
        gt_next_2 = [wt_pred_next_label]*adata_wt.shape[0]
        gt_next = np.concatenate([gt_next_1,gt_next_2])
        adata_bwmerge.obs[ 'genotype_next'] = gt_next

        # feed into model
        model.to(device)
        eval_results_0 = eval_testdata(model, adata_bwmerge, gene_ids,
                                    train_data_dict={"cell_type_to_index":cell_type_to_index,
                                                      "genotype_to_index":genotype_to_index,
                                                      "vocab":vocab,},
                                    config = config,
                                    make_plots=False)
        #
        a_eva=eval_results_0['adata']
        cell_emb_data=a_eva.obsm['X_scGPT_next'] #[:10000,:]
        perturb_info=a_eva.obs[['genotype','genotype_next']]

        perturb_info['round']=n_round
        perturb_info['type']=['pert_source']*adata_target.shape[0]*n_expands_per_epoch + ['pert_dest']*adata_wt.shape[0]

        # calculate cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity

        # Calculate cosine similarity
        #adata_eva_copy_othergenes
        cell_emb_data_treat = cell_emb_data[perturb_info['type']=='pert_source']

        cell_emb_data_ref = cell_emb_data[perturb_info['type']=='pert_dest']
        # Calculate cosine similarity
        cosine_sim_matrix = cosine_similarity(cell_emb_data_ref, cell_emb_data_treat)
        cs_matrix_res[:,:,n_round] = cosine_sim_matrix
        if cell_emb_data_all is None:
            cell_emb_data_all = cell_emb_data
            perturb_info_all = perturb_info

        else:
            cell_emb_data_all = np.concatenate([cell_emb_data_all, cell_emb_data], axis=0)
            perturb_info_all = pd.concat([perturb_info_all, perturb_info], axis=0)

    perturb_info_all.reset_index(inplace=True)
    return cell_emb_data_all, perturb_info_all, cs_matrix_res, a_eva


def calculate_avg_cosine_similarity(input_mat,pd_pert_f):
    """
    Calculate average cosine similarity from a given cosine similarity matrix of size (n_ctrl,n_cell,n_round),
     and a pandas dataframe pd_pert_tf of size (nctrl+n_cell)*n_round
    """
    # Calculate cosine similarity
    cs_nx1 = np.sum(input_mat, axis=0)/input_mat.shape[0]


    # prompt: convert cs_nx1 to a 1-dimension array

    cs_nx1_1d = cs_nx1.flatten('F')

    perturb_f_p=pd_pert_f[ pd_pert_f['type'] == 'pert_source']
    perturb_f_p.shape

    perturb_f_p['cosine_sim_matrix_column_avg'] = cs_nx1_1d
    perturb_f_p.reset_index(inplace=True)
    return perturb_f_p
