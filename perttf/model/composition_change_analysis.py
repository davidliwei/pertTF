import torch
import scanpy as sc
import numpy as np
from perttf.model.train_function import eval_testdata

def calculate_lonESS_score(adata, overall_fraction_dict=None,
                           recalculate_nn=False,
                           nn_name=None,
                           n_neighbors=30,
                           target_genotype = None,
                           n_pcs=20,
                           delta = 0.0001,
                           ):
  """
  calculate the lochNESS score for a single cell
  """
  n_cells = adata.n_obs
  # calculate the overall fraction
  if overall_fraction_dict is None:
    overall_fraction = adata.obs['genotype'].value_counts(normalize=True)
    # convert overall_fraction into a dictionary structure
    overall_fraction_dict = overall_fraction.to_dict()

  if recalculate_nn:
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs,key_added=nn_name)
  nn_key='distances' if nn_name is None else f'{nn_name}_distances'
  # iterate each cell

  lowess_vec=[]
  for cell_id in adata.obs_names:
    cell_index = adata.obs_names.get_loc(cell_id)
    if target_genotype is not None:
      cell_genotype=target_genotype
    else:
      cell_genotype=adata.obs.loc[cell_id,'genotype']
    neighboring_vec = adata.obsp[nn_key][cell_index, :]
    indices = neighboring_vec.nonzero()[1]
    neighboring_cells = adata.obs.index[indices]
    neighboring_genotypes = sum(adata.obs.loc[neighboring_cells, 'genotype'] == cell_genotype)
    lonESS_score = neighboring_genotypes / n_neighbors
    overall_score = overall_fraction_dict[cell_genotype]

    #loness_score_adjusted = lonESS_score - overall_score
    if overall_score ==0:
      overall_score = 0.0001
    lonESS_score_adjusted = lonESS_score / overall_score -1
    # generate a random noise and added to the score
    if delta > 0:
      noise = np.random.normal(0, delta)
      lonESS_score_adjusted += noise
    lowess_vec.append(lonESS_score_adjusted)
  return lowess_vec

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
    adata_bwmerge=sc.concat([adata_wt]*n_expands_per_epoch,axis=0)
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
        eval_results_0 = eval_testdata(model, adata_bwmerge, gene_ids,
                                    train_data_dict={"cell_type_to_index":cell_type_to_index,
                                                      "genotype_to_index":genotype_to_index,
                                                      "vocab":vocab,},
                                    config = config,
                                    make_plots=False)
        #
        a_eva=eval_results_0 #['adata']
        pred_ps_score = a_eva.obsm['ps_pred_next']
        perturb_info=a_eva.obs[['genotype','genotype_next']]

        perturb_info['round']=n_round
        perturb_info['pred_ps'] = pred_ps_score
        #perturb_info['type']=['pert_source']*adata_target.shape[0]*n_expands_per_epoch + ['pert_dest']*adata_wt.shape[0]

        if perturb_info_all is None:
            perturb_info_all = perturb_info

        else:
            perturb_info_all = pd.concat([perturb_info_all, perturb_info], axis=0)

    perturb_info_all.reset_index(inplace=True)
    return  perturb_info_all,  a_eva
