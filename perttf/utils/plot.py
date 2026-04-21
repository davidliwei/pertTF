import multiprocessing
import scanpy as sc
import matplotlib.pyplot as plt
import wandb
import anndata
import os
from pathlib import Path

from scipy.stats import pearsonr
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    accuracy_score,
    average_precision_score, # For AUPR
)
from sklearn.preprocessing import label_binarize

def plot_confusion_matrix(y_true, y_pred, class_labels, title):
    """Generates and plots a confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d", # Integer format
        cmap="Blues",
        xticklabels=class_labels,
        yticklabels=class_labels,
    )
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title(title)
    plt.show()

def get_classification_metrics(
    adata,
    task_name: str,
    metrics_to_log: dict = {},
    metrics_prefix: str='test/'
):
    """
    Calculates and displays comprehensive classification metrics for a multi-class task.

    Args:
        adata: Your AnnData object.
        true_label_col: The column name in adata.obs for the ground truth labels.
        pred_label_col: The column name in adata.obs for the predicted labels.
        proba_obsm_key: The key in adata.obsm where the (n_obs, n_classes)
                        prediction probabilities are stored.
    """
    true_label_col=task_name
    true_label_col_id=f"{task_name}_id"
    pred_label_col=f"predicted_{task_name}"
    proba_obsm_key=f"{task_name}_pred_probs"
    df = adata.obs
    obsm = adata.obsm
    y_true = df[true_label_col]
    y_pred = df[pred_label_col]
    y_true_id = df[true_label_col_id]
    # Get the unique class labels in the correct order
    class_labels = sorted(y_true.unique())
    class_labels_id = sorted(y_true_id.unique().categories)
    # 1. Classification Report (Precision, Recall, F1-Score)
    # --- Metrics requiring probabilities ---
    if proba_obsm_key in adata.obsm:
        y_probas = obsm[proba_obsm_key]
        # Ensure y_probas columns align with class_labels.
        # This is a common source of error. The example assumes the
        # model's output probability columns are in the sorted order of class names.
        
        # Binarize the true labels for multi-class AUC calculations
        y_true_binarized = label_binarize(y_true_id, classes=class_labels_id)

        # 2. ROC-AUC Score (One-vs-Rest)
        # We use a weighted average to account for class imbalance.
        roc_auc = roc_auc_score(
            y_true_binarized,
            y_probas,
            multi_class="ovr",
            average="macro"
        )
        f1 = f1_score(
            y_true,
            y_pred,
            average="macro"
        )
        accuracy = accuracy_score(y_true, y_pred)
        # 3. AUPR Score (Area Under Precision-Recall Curve)
        aupr = average_precision_score(y_true_binarized, y_probas, average="macro")


    else:
        print(f"'{proba_obsm_key}' not found in adata.obsm. Skipping ROC-AUC and AUPR calculation.\n")

    # 4. Confusion Matrix Plot
    #plot_confusion_matrix(
        #y_true,
        #y_pred,
        #class_labels=class_labels,
        #title=f"Confusion Matrix for '{true_label_col}'"
    #)
    
    metrics_to_log[f"{metrics_prefix}_{true_label_col}_auc"] = roc_auc
    metrics_to_log[f"{metrics_prefix}_{true_label_col}_aupr"] = aupr 
    metrics_to_log[f"{metrics_prefix}_{true_label_col}_f1"] = f1
    metrics_to_log[f"{metrics_prefix}_{true_label_col}_acc"] = accuracy
    return metrics_to_log


def expression_correlation( adata, 
                            expr_layer = 'next_expr', 
                            pred_layer = 'mvc_next_expr', 
                            zero_layer = None,
                            metrics_to_log: dict = {},
                            metrics_prefix: str='test/'):
    true_expr = adata.layers[expr_layer].toarray()
    pred_expr = adata.obsm[pred_layer]
    true_expr_mean = true_expr.mean(0)
    pred_expr_zero = adata.obsm[zero_layer] if zero_layer is not None else 1
    pred_expr = pred_expr * pred_expr_zero
    pred_corr = np.mean([pearsonr(pred_expr[i,:], true_expr[i,:])[0] for i in range(true_expr.shape[0])])
    mean_corr = np.mean([pearsonr(true_expr_mean, true_expr[i,:])[0] for i in range(true_expr.shape[0])])
    metrics_to_log[f"{metrics_prefix}_pred_next_corr"] = pred_corr
    metrics_to_log[f"{metrics_prefix}_mean_expr_corr"] = mean_corr
    return metrics_to_log


def process_and_log_umaps(adata_t, config, epoch: int, eval_key: str, save_dir: Path, data_gen_ps_names: list = None):
    """
    Worker function to run UMAP, plotting, and logging in a separate process.
    """
    try:
        print(f"[Process {os.getpid()}] Starting UMAP and plotting for epoch {epoch}, key '{eval_key}'.")
        # Load the AnnData object from the provided path
        #adata_t 
        # This block is moved directly from your original `eval_testdata` function
        results = {}
        metrics_to_log = {"epoch": epoch}
        if 'mvc_next_expr' in adata_t.obsm.keys() and 'next_expr' in adata_t.layers.keys() and config.next_cell_pred_type == 'pert':
            #print('start perturbation expression eval')
            adata_wt = adata_t[adata_t.obs.genotype == 'WT',:]
            metrics_to_log = expression_correlation(adata_wt, 
                                                    expr_layer = 'next_expr', 
                                                    pred_layer = 'mvc_next_expr', 
                                                    metrics_to_log = metrics_to_log,
                                                    metrics_prefix = f'test/{eval_key}')
        #print('start class eval')
        cls_tasks = []
        if config.get('cell_type_classifier', True) and config.cell_type_classifier_weight > 0:
            cls_tasks.append('celltype')
        if config.get('genotype_classifier', True) and config.perturbation_classifier_weight > 0:
            cls_tasks.append('genotype')
        for task in cls_tasks:
            metrics_to_log = get_classification_metrics(adata_t,
                                                    task_name=task,
                                                    metrics_to_log = metrics_to_log,
                                                    metrics_prefix = f'test/{eval_key}')
        
        print('start umap')
        if config.next_cell_pred_type == 'pert':
            sc.pp.neighbors(adata_t, use_rep="X_scGPT_next")
            sc.tl.umap(adata_t, min_dist=0.3)
            if config.cell_type_classifier_weight > -1:
                fign1 = sc.pl.umap(adata_t, color=["celltype"],
                    title=[f"{eval_key} celltype, e{epoch}, pred embedding",],
                    frameon=False, return_fig=True, show=False)
                results["next_umap_celltype"] = fign1
            if config.perturbation_classifier_weight > -1:
                fign2 = sc.pl.umap(adata_t, color=["genotype"],
                    title=[f"{eval_key} genotype, e{epoch}, pred embedding",],
                    frameon=False, return_fig=True, show=False)
                results["next_umap_genotype"] = fign2
                fign3 = sc.pl.umap(adata_t, color=["genotype_next"],
                    title=[f"{eval_key} next genotype, e{epoch}, pred embedding",],
                    frameon=False, return_fig=True, show=False)
                results["next_umap_genotype_next"] = fign3

        sc.pp.neighbors(adata_t, use_rep="X_scGPT")
        sc.tl.umap(adata_t, min_dist=0.3)

        if "batch" in adata_t.obs:
            fig = sc.pl.umap(adata_t, color=["batch"], title=[f"{eval_key} batch, e{epoch}"],
                frameon=False, return_fig=True, show=False)
            results["batch_umap"] = fig

        if config.cell_type_classifier_weight > -1:
            fig = sc.pl.umap(adata_t, color=["celltype"], title=[f"{eval_key} celltype, e{epoch}"],
                frameon=False, return_fig=True, show=False)
            results["celltype_umap"] = fig
            if config.get('cell_type_classifier', True):
                fig4 = sc.pl.umap(adata_t, color=["predicted_celltype"], title=[f"{eval_key} pred celltype, e{epoch}"],
                    frameon=False, return_fig=True, show=False)
                results["pred_celltype"] = fig4

        if config.perturbation_classifier_weight > -1:
            fig = sc.pl.umap(adata_t, color=["genotype"], title=[f"{eval_key} genotype, e{epoch}"],
                frameon=False, return_fig=True, show=False)
            results["genotype_umap"] = fig
            if config.get('genotype_classifier', True):
                fig3 = sc.pl.umap(adata_t, color=["predicted_genotype"], title=[f"{eval_key} pred genotype, e{epoch}"],
                    frameon=False, return_fig=True, show=False)
                results["pred_genotype"] = fig3
            if "genotype_next" in adata_t.obs:
                fig5 = sc.pl.umap(adata_t, color=["genotype_next"], title=[f"{eval_key} next genotype, e{epoch}"],
                    frameon=False, return_fig=True, show=False)
                results["genotype_next"] = fig5
        
        # Save images and prepare for wandb logging
        save_image_types = [
            "batch_umap", "celltype_umap", "genotype_umap", "pred_genotype",
            "pred_celltype", "genotype_next", "next_umap_celltype",
            "next_umap_genotype", "next_umap_genotype_next"
        ]
        saved_images = {}
        for res_key, res_img_val in results.items():
            if res_key in save_image_types:
                save_path = save_dir / f"{eval_key}_embeddings_{res_key}_e{epoch}.png"
                res_img_val.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close(res_img_val) # Close the figure to free memory
                saved_images[f"test/{eval_key}_{res_key}"] = str(save_path)
                #wandb.Image(str(save_path), caption=f"{eval_key}_{res_key} epoch {epoch}")
        
        # Handle Loness score plotting
        if config.ps_weight > 0:
            loness_columns = [x for x in adata_t.obs if x.startswith('lonESS')]
            for lon_c in loness_columns:
                fig_lonc = sc.pl.umap(adata_t, color=[lon_c], title=[f"loness {lon_c} e{epoch}"],
                    frameon=False, return_fig=True, show=False)
                lon_c_rep = lon_c.replace('/', '_')
                fig_lonc.savefig(save_dir / f"{eval_key}_loness_{lon_c_rep}_e{epoch}.png", dpi=300, bbox_inches='tight')
                plt.close(fig_lonc)

            if data_gen_ps_names is not None and 'ps_pred' in adata_t.obsm:
                predicted_ps_score = adata_t.obsm['ps_pred']
                for si_i, lon_c in enumerate(data_gen_ps_names):
                    lon_c_rep = lon_c.replace('/', '_')
                    adata_t.obs[f'{lon_c_rep}_pred'] = predicted_ps_score[:, min(si_i, predicted_ps_score.shape[1]-1)]
                    fig_lonc_pred = sc.pl.umap(adata_t, color=[f'{lon_c_rep}_pred'], title=[f"loness {lon_c_rep}_pred e{epoch}"],
                        frameon=False, return_fig=True, show=False)
                    fig_lonc_pred.savefig(save_dir / f"{eval_key}_loness_{lon_c_rep}_pred_e{epoch}.png", dpi=300, bbox_inches='tight')
                    plt.close(fig_lonc_pred)

        # Log all collected metrics to wandb
        #if metrics_to_log:
            #wandb.log(metrics_to_log)
        # added: write validation adata_t back to disk
        if hasattr(config, "save_validation_h5ad") and config.save_validation_h5ad:
            adata_t.write_h5ad(save_dir / f'adata_last_validation_{eval_key}.h5ad')
        return {
            'images': saved_images,
            'metrics': metrics_to_log,
            'eval_dict_key': eval_key,
            'epoch': epoch
        }
        


    except Exception as e:
        print(f"Error in background UMAP process: {e}")
    #finally:
     #   # Clean up the temporary AnnData file
      #  if os.path.exists(adata_path):
       #     os.remove(adata_path)
