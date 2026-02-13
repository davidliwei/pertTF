import numpy as np
import jax
import jax.numpy as jnp
from ott.geometry import pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn
from functools import partial

# --- JAX Solver with Padding Support ---
# 1. epsilon is REMOVED from static_argnames (Fixes the main recompilation bug)
# 2. top_k remains static because it determines output shape size
@partial(jax.jit, static_argnames=['top_k'])
def _solve_ot_padded_jit(
    X_source_padded, 
    X_target_padded, 
    weights_source, 
    weights_target, 
    epsilon, 
    top_k=10
    ):
    """
    Solves OT with padded inputs. 
    Weights of 0.0 indicate padded (fake) points to be ignored by Sinkhorn.
    """
    # Define Geometry
    geom = pointcloud.PointCloud(X_source_padded, X_target_padded, epsilon=epsilon)
    
    # Define Problem with weights (a=source weights, b=target weights)
    # Sinkhorn handles 0-weights by treating them as having no mass.
    prob = linear_problem.LinearProblem(geom, a=weights_source, b=weights_target)
    
    # Solve
    out = sinkhorn.Sinkhorn()(prob)
    P = out.matrix
    
    # Get Top-K Probabilities and Indices
    weights, local_indices = jax.lax.top_k(P, k=top_k)
    
    # Normalize weights (avoid division by zero for padded rows)
    sum_weights = jnp.sum(weights, axis=1, keepdims=True) + 1e-10
    weights = weights / sum_weights
    
    # Compute Distances
    target_points = X_target_padded[local_indices]
    source_points = X_source_padded[:, None, :]
    dists = jnp.sum((source_points - target_points) ** 2, axis=-1)
    
    return weights, local_indices, dists

def _get_next_power_of_2(n, min_size=128):
    """
    Returns the next power of 2 greater than or equal to n.
    Clamped at a minimum size to avoid tiny kernels.
    """
    if n <= min_size:
        return min_size
    # This bit-shifting trick finds the next power of 2
    return 1 << (n - 1).bit_length()

def _get_padded_arrays(X, min_size=128, X_weights = None):
    """Pads array X to the next power of 2."""
    n_rows, n_cols = X.shape
    
    # CALCULATE NEW SHAPE
    n_pad = _get_next_power_of_2(n_rows, min_size=min_size)
    
    # Pad Data with zeros
    X_pad = np.zeros((n_pad, n_cols))
    X_pad[:n_rows] = X
    
    # Create Weights
    # 1.0 for real data, 0.0 for padded
    weights = np.zeros(n_pad)
    # Normalize weights so they sum to 1 relative to the REAL data
    if X_weights != None:
        assert X_weights.shape[0] == X.shape[0], f'Mismatch shapte between X and X_weights, {X_weights.shape} vs {X.shape}'
        weights[:n_rows] = X_weights/np.sum(X_weights)
    else:
        weights[:n_rows] = 1.0 / n_rows
    
    return X_pad, weights, n_rows

def compute_ot_for_subset(
        adata_subset, 
        top_k=10, 
        epsilon='auto', 
        max_dist_sq="auto", 
        red_key='X_pca', 
        epsilon_scaler=0.01, 
        min_bucket=128
    ):
    ot_results = {}
    
    # Iterate by Cell Type
    unique_celltypes = adata_subset.obs['celltype'].unique()
    
    for ctype in unique_celltypes:
        mask_ctype = adata_subset.obs['celltype'] == ctype
        
        # Identify Source (WT)
        mask_wt = mask_ctype & (adata_subset.obs['genotype'] == 'WT')
        if not np.any(mask_wt): continue 

        # Prepare Source Data
        X_wt_raw = adata_subset.obsm[red_key][mask_wt]
        wt_names = adata_subset.obs.index[mask_wt].tolist()
        
        # Pad Source ONCE per cell type
        X_wt_pad, w_wt, n_wt = _get_padded_arrays(X_wt_raw, min_size=min_bucket)
        #print('-------------------------')
        #print(f'wt {ctype} bucket size: {X_wt_pad.shape}')
        X_wt_jax = jnp.array(X_wt_pad)
        w_wt_jax = jnp.array(w_wt)

        # Iterate over Perturbations
        available_perts = adata_subset.obs.loc[mask_ctype, 'genotype'].unique()
        available_perts = [p for p in available_perts if p != 'WT']
        
        for pert in available_perts:
            mask_pert = mask_ctype & (adata_subset.obs['genotype'] == pert)
            if mask_pert.sum() < top_k: continue

            # Prepare Target Data
            X_pert_raw = adata_subset.obsm[red_key][mask_pert]
            target_names = adata_subset.obs.index[mask_pert].values 
            
            # Pad Target
            X_pert_pad, w_pert, n_pert = _get_padded_arrays(X_pert_raw, min_size=min_bucket)
            #print(f'{pert} {ctype} bucket size: {X_pert_pad.shape}')
            X_pert_jax = jnp.array(X_pert_pad)
            w_pert_jax = jnp.array(w_pert)
            
            # --- DYNAMIC THRESHOLD ---
            current_threshold = max_dist_sq if max_dist_sq is not None else np.inf
            # Use raw (unpadded) numpy arrays for estimation
            auto_threshold, auto_eps = estimate_context_threshold(X_wt_raw, X_pert_raw, epsilon_scaler=epsilon_scaler)
            current_threshold = auto_threshold if max_dist_sq == "auto" else current_threshold
            this_epsilon = auto_eps if epsilon == 'auto' else epsilon
            
            # --- EXECUTE JAX (Padded) ---
            try:
                # Pass epsilon as a regular argument, not static
                weights, indices, dists = _solve_ot_padded_jit(
                    X_wt_jax, X_pert_jax, w_wt_jax, w_pert_jax, 
                    epsilon=this_epsilon, top_k=top_k
                )
                
                # Move to CPU and Slice off padding
                # We only need the first n_wt rows (real source cells)
                weights_np = np.array(weights[:n_wt])
                indices_np = np.array(indices[:n_wt])
                dists_np = np.array(dists[:n_wt])
                
                # --- FORMAT RESULTS ---
                for i, wt_name in enumerate(wt_names):
                    if dists_np[i, 0] > current_threshold:
                        continue 

                    if wt_name not in ot_results: ot_results[wt_name] = {}
                    
                    # Safety check: ensure indices don't point to padded target zones
                    # (Sinkhorn *shouldn't* pick them because weight is 0, but top_k might grab them if K > valid targets)
                    valid_match_mask = indices_np[i] < n_pert
                    
                    # If we somehow picked a padded cell, we filter it out here
                    chosen_indices = indices_np[i][valid_match_mask]
                    chosen_weights = weights_np[i][valid_match_mask]
                    
                    chosen_target_names = target_names[chosen_indices]
                    ot_results[wt_name][pert] = (chosen_target_names, chosen_weights)
                    
            except Exception as e:
                print(f"OT Failed for {ctype} -> {pert}: {e}")
                
    return ot_results

def estimate_context_threshold(X_wt, X_pert, epsilon_scaler = 0.01, sample_size=1000):
    """
    Estimates a distance threshold specifically for this WT -> Pert pair.
    We compute the median distance between random pairs of (WT, Pert) cells.
    """
    from scipy.spatial.distance import cdist
    
    # Subsample if clouds are too large to speed up cdist
    if X_wt.shape[0] > sample_size:
        idx_wt = np.random.choice(X_wt.shape[0], sample_size, replace=False)
        X_wt_sub = X_wt[idx_wt]
    else:
        X_wt_sub = X_wt
        
    if X_pert.shape[0] > sample_size:
        idx_pert = np.random.choice(X_pert.shape[0], sample_size, replace=False)
        X_pert_sub = X_pert[idx_pert]
    else:
        X_pert_sub = X_pert
        
    # Calculate cross-distances (WT vs Pert)
    # cdist returns matrix of size (N_wt x N_pert)
    dists = cdist(X_wt_sub, X_pert_sub, metric='sqeuclidean')
    median_dist = np.median(dists)
    
    # Heuristic: Set epsilon to ~5-10% of the median squared distance
    # This ensures the exponent -dist/eps is roughly -10 to -20, preventing numerical collapse
    suggested_epsilon = median_dist * epsilon_scaler
    
    return median_dist, suggested_epsilon