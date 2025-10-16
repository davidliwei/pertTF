import numpy as np
from typing import Literal

# Courtesy of Gemini AI, unbin predicted expr values if they were binned prior to scGPT
def unbin_matrix(
    binned_matrix: np.ndarray,
    bin_edges_matrix: np.ndarray,
    method: Literal["mean", "median"] = "mean",
) -> np.ndarray:
    """
    Converts a 2D matrix of binned indices back to continuous values.

    This function processes a matrix where each row represents a cell with
    integer bin indices. It uses a corresponding matrix of bin edges, where
    each row defines the bin boundaries for the respective cell.

    Args:
        binned_matrix (np.ndarray):
            A 2D numpy array of integer bin indices (cells x genes).
        bin_edges_matrix (np.ndarray):
            A 2D numpy array of bin edge values. Each row i corresponds to
            the bin edges for row i in binned_matrix. Note that rows can
            have different numbers of bins and therefore different lengths.
            This is handled by providing it as a numpy array of objects
            (dtype=object).
        method (Literal["mean", "median"], optional):
            The method to estimate the unbinned value. For a uniform
            distribution within a bin, the mean and median are the same.
            Defaults to "mean".

    Returns:
        np.ndarray:
            A 2D numpy array of the same shape as `binned_matrix` with the
            approximated continuous float values.
    """
    # --- Input Validation ---
    if binned_matrix.ndim != 2:
        raise ValueError("Input `binned_matrix` must be a 2D numpy array.")
    # For bin_edges_matrix, we expect an array of arrays, so we check the first dimension
    if binned_matrix.shape[0] != len(bin_edges_matrix):
        raise ValueError(
            "The number of rows in `binned_matrix` must match the number of "
            "rows in `bin_edges_matrix`."
        )

    # --- Initialization ---
    unbinned_matrix = np.zeros_like(binned_matrix, dtype=float)

    # --- Processing ---
    for i in range(binned_matrix.shape[0]):
        binned_row = binned_matrix[i]
        bin_edges_row = bin_edges_matrix[i]
        
        # Get unique non-zero bin indices present in the current row
        unique_bins = np.unique(binned_row[binned_row > 0])

        for bin_idx in unique_bins:
            if bin_idx >= len(bin_edges_row):
                raise IndexError(
                    f"In row {i}, bin index {bin_idx} is out of bounds for the "
                    f"provided bin_edges array of length {len(bin_edges_row)}."
                )

            lower_bound = bin_edges_row[bin_idx - 1]
            upper_bound = bin_edges_row[bin_idx]

            if method in ["mean", "median"]:
                rep_value = (lower_bound + upper_bound) / 2.0
            else:
                raise ValueError("Method must be either 'mean' or 'median'.")
            
            # Assign the representative value to all matching locations in the row
            unbinned_matrix[i, binned_row == bin_idx] = rep_value

    return unbinned_matrix