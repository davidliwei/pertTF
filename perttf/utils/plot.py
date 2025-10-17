import multiprocessing
import scanpy as sc
import matplotlib.pyplot as plt
import wandb
import anndata
import os
from pathlib import Path

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

        if config.next_cell_pred_type == 'pert':
            sc.pp.neighbors(adata_t, use_rep="X_scGPT_next")
            sc.tl.umap(adata_t, min_dist=0.3)
            if config.cell_type_classifier:
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

        if config.cell_type_classifier:
            fig = sc.pl.umap(adata_t, color=["celltype"], title=[f"{eval_key} celltype, e{epoch}"],
                frameon=False, return_fig=True, show=False)
            results["celltype_umap"] = fig
            fig4 = sc.pl.umap(adata_t, color=["predicted_celltype"], title=[f"{eval_key} pred celltype, e{epoch}"],
                frameon=False, return_fig=True, show=False)
            results["pred_celltype"] = fig4

        if config.perturbation_classifier_weight > -1:
            fig = sc.pl.umap(adata_t, color=["genotype"], title=[f"{eval_key} genotype, e{epoch}"],
                frameon=False, return_fig=True, show=False)
            results["genotype_umap"] = fig
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

        return {
            'images': saved_images,
            'metrics': metrics_to_log,
            'eval_dict_key': eval_key,
            'epoch': epoch
        }
        
        # added: write validation adata_t back to disk
        # adata_t.write_h5ad(save_dir / f'adata_last_validation_{eval_key}.h5ad')

    except Exception as e:
        print(f"Error in background UMAP process: {e}")
    #finally:
     #   # Clean up the temporary AnnData file
      #  if os.path.exists(adata_path):
       #     os.remove(adata_path)
