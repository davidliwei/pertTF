
import torch
import numpy as np
from typing import Literal
import matplotlib.pyplot as plt

from perttf.model.pert_emb import generate_pert_embeddings, calculate_avg_cosine_similarity
from perttf.model.train_function import eval_testdata


def virtual_screen_src2dest(adata_src,
                            adata_des,
                            model, vocab, config,
                            running_parameters,
                            screen_genes=None,
                            device=None,
                            n_expands_per_epoch = 5,
                            n_epoch = 10,
                            wt_pred_next_label = 'WT',
                            ):
  """
  A function to perform virtual screen from source to destination.

  """

  cell_type_to_index = running_parameters['cell_type_to_index']
  genotype_to_index = running_parameters['genotype_to_index']
  #genes = running_parameters['genes']
  gene_ids = running_parameters['gene_ids']

  # default: device
  if device is None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # default: screen_genes
  gene_available_for_screen=genotype_to_index.keys()
  if screen_genes is None:
    screen_genes_next=gene_available_for_screen
  else:
    filtered_genes=[x for x in screen_genes if x not in gene_available_for_screen]
    if len(filtered_genes)>0:
      print("warning: the following genes are filtered (not in genotype index; cannot screen):" + ','.join(filtered_genes))
    screen_genes_next=[x for x in screen_genes if x in gene_available_for_screen]
    if len(screen_genes_next)==0:
      raise ValueError('None of the genes are screenable')

  print(f"genes to screen: {len(screen_genes_next)}" )
  # generate perturbation embeddings
  cell_emb_all, perturb_f_all, cs_mat_res, adata_last_eva = generate_pert_embeddings(adata_src,
                                                                        adata_des,
                                                                        screen_genes_next,
                                                                        model, gene_ids,
                                                                        cell_type_to_index, genotype_to_index,
                                                                        vocab, config, device,
                                                                        n_expands_per_epoch= n_expands_per_epoch,
                                                                        n_epoch= n_epoch,
                                                                        wt_pred_next_label =   wt_pred_next_label,  )
  # calculate avg cosine similarity
  perturb_f_p = calculate_avg_cosine_similarity(cs_mat_res, perturb_f_all)
  # calculate statiscis and sort

  average_cosine_similarity = perturb_f_p.groupby('genotype_next')['cosine_sim_matrix_column_avg'].agg(
      mean='mean',
      median='median',
      count='count',
      percentile_90=lambda x: np.percentile(x, 90),)

  # Sort the result
  sorted_average_cosine_similarity = average_cosine_similarity.sort_values(by='mean',ascending=False)

  # Add ranking
  sorted_average_cosine_similarity = sorted_average_cosine_similarity.reset_index()
  sorted_average_cosine_similarity['rank'] = sorted_average_cosine_similarity['mean'].rank(ascending=False, method='dense')
  sorted_average_cosine_similarity['rank_median'] = sorted_average_cosine_similarity['median'].rank(ascending=False, method='dense')
  sorted_average_cosine_similarity['rank_percentile90'] = sorted_average_cosine_similarity['percentile_90'].rank(ascending=False, method='dense')

  sorted_average_cosine_similarity.set_index('genotype_next', inplace=True)
  sorted_average_cosine_similarity = sorted_average_cosine_similarity.sort_values(by='percentile_90',ascending=False)

  ret_dict={'sim_ranked':sorted_average_cosine_similarity,
            'adata_last_eva':adata_last_eva}
  return ret_dict

def plot_predicted_eva(a_eva,subset_frac=1.0,
                       redo_umap=True,
                       umap_use_rep: Literal["X_scGPT_next","X_scGPT"] = "X_scGPT_next",
                       show=True,):
  """
  A function to plot predicted evaluations from perturbed embeddings
  """
  #sc.pp.neighbors(adata, use_rep="X_scGPT_next")
  randsel_ss=np.random.random(a_eva.shape[0])
  import scanpy as sc
  a_eva_cp=a_eva [randsel_ss <= subset_frac]# .copy()
  results={}
  if redo_umap:
    sc.pp.pca(a_eva_cp, layer=umap_use_rep)
    sc.pp.neighbors(a_eva_cp, use_rep=umap_use_rep)
    sc.tl.umap(a_eva_cp, min_dist=0.5)

  if 'celltype' in a_eva_cp.obs:
    t_celltype=sc.pl.umap(a_eva_cp, color=["celltype"],
        title=[f"celltype, pred embedding",],
        frameon=False,
        return_fig=True,
        palette="tab20b",
        show=show,
        #legend_loc=None,
    )
    results['celltype']=t_celltype
    plt.close(t_celltype)
  if ('genotype' not in a_eva_cp.obs) or ('genotype_next' not in a_eva_cp.obs) :
    raise ValueError('adata_eva must have genotype and genotype_next columns')

  t_genotype=sc.pl.umap(a_eva_cp, color=["genotype"],
      title=[f"genotype, pred embedding",],
      frameon=False,
      return_fig=True,
      palette="tab20b",
      show=show,
      #legend_loc=None,
  )
  results['genotype']=t_genotype
  plt.close(t_genotype)

  t_genotype_next=sc.pl.umap(a_eva_cp, color=["genotype_next"],
      title=[f"genotype_next, pred embedding",],
      frameon=False,
      return_fig=True,
      palette="tab20b",
      show=show,
      #legend_loc=None,
  )
  results['genotype_next']=t_genotype_next
  plt.close(t_genotype_next)
  #cm = plt.get_cmap('tab20b')
  #NUM_COLORS=30

  # prompt: concatenate both genotype and genotype_next columns of a_eva_cp.obs into a new column named "genotype_combined"

  a_eva_cp.obs["genotype_combined"] = a_eva_cp.obs["genotype"].astype(str) + "->" + a_eva_cp.obs["genotype_next"].astype(str)
  t_genotype_combined=sc.pl.umap(a_eva_cp, color=["genotype_combined"],
      title=[f"genotype_next, pred embedding",],
      frameon=False,
      return_fig=True,
      #palette="tab20b",
      palette="Set1",
      show=show,
      #palette=[cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)],
      #legend_loc=None,
  )
  results['genotype_combined']=t_genotype_combined
  plt.close(t_genotype_combined)

  t_genotype_combined_pca=sc.pl.pca(a_eva_cp, color=["genotype_combined"],
      title=[f"genotype_next, pred embedding",],
      frameon=False,
      return_fig=True,
      #palette="tab20b",
      palette="Set1",
      show=show,
      #palette=[cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)],
      #legend_loc=None,
  )
  results['genotype_combined_pca']=t_genotype_combined_pca
  plt.close(t_genotype_combined_pca)

  if "ps_pred_next" in a_eva_cp.obs:
    t_ps_pred_next=sc.pl.umap(a_eva_cp, color=["ps_pred_next"],
        title=[f"ps_pred_next",],
        frameon=False,
        return_fig=True,
        #palette="tab20b",
        show=show,
        #legend_loc=None,
    )
    results['ps_pred_next']=t_ps_pred_next
    plt.close(t_ps_pred_next)
    #sc.pl.umap(a_eva_cp, color=["ps_pred_next"],
    ##    groups  = "genotype_combined",
    #    title=[f"ps_pred_next, grouped",],
    #    frameon=False,
    #    return_fig=False,
    #    #palette="tab20b",
    #    show=show,
    #    #legend_loc=None,
    #)
  results['adata']=a_eva_cp
  return results
  #return a_eva_cp

def generate_pred_ps(adata_src,
                    model, vocab, config,
                    running_parameters,
                    target_pred_gene_list=[],
                    device=None,
                    addwt=True, wtlabel='WT',
                    use_wt_only = True,
                    ):
  """
  Generate predicted PS values
  Parameters:
    addwt: bolean, whether to add additional "WT" label (defined in wt_label) to the target_pred_gene_list
    wtlabel: the label in "genotype" column used to identify "wild-type" cells
    use_wt_only: bollean, only simulate perturbations in WT cells
  """

  cell_type_to_index = running_parameters['cell_type_to_index']
  genotype_to_index = running_parameters['genotype_to_index']
  #genes = running_parameters['genes']
  gene_ids = running_parameters['gene_ids']

  adata_src.obs['genotype_next']= adata_src.obs['genotype']  # 'PDX1' # set the next predicted

  # Ensure that 'WT' and target_pred_gene are in the categories of 'genotype_next'
  new_categories = list(adata_src.obs['genotype_next'].cat.categories)
  if addwt and wtlabel not in target_pred_gene_list:
      target_pred_gene_list.append(wtlabel)
  #if 'WT' not in new_categories:
  #    new_categories.append('WT')
  for target_pred_gene in target_pred_gene_list:
    if target_pred_gene not in new_categories:
        new_categories.append(target_pred_gene)
  adata_src.obs['genotype_next'] = adata_src.obs['genotype_next'].cat.set_categories(new_categories)

  if use_wt_only:
    adata_src.obs.loc[adata_src.obs['genotype'].isin([wtlabel]), 'genotype_next'] \
        = np.random.choice(target_pred_gene_list, size = sum(adata_src.obs['genotype'].isin([wtlabel])))
  else:
    adata_src.obs[ 'genotype_next'] \
    = np.random.choice(target_pred_gene_list, size = adata_src.n_obs)
  #adata_src.obs.loc[adata_src.obs['genotype']=='WT' ,'genotype_next'].value_counts()

  model.to(device)
  eval_results = eval_testdata(model, adata_src,gene_ids,
                              train_data_dict={"cell_type_to_index":cell_type_to_index,
                                                "genotype_to_index":genotype_to_index,
                                                "vocab":vocab,},
                              config = config)

  adata_eva=eval_results #['adata']
  adata_eva
  adata_eva.obs['ps_pred_next']=adata_eva.obsm['ps_pred_next']

  return adata_eva
