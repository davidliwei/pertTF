# This is modified from code from scGPT's gene_tokenizer to accomodate tokenizing next_cell pert target data
import json
import pickle
from pathlib import Path
from collections import Counter, OrderedDict
from typing import Dict, Iterable, List, Optional, Tuple, Union
from typing_extensions import Self

import numpy as np
import pandas as pd
import torch
import torchtext.vocab as torch_vocab
from torchtext.vocab import Vocab

# from transformers.tokenization_utils import PreTrainedTokenizer
# from transformers import AutoTokenizer, BertTokenizer


# This function remains unchanged
def tokenize_batch(
    data: np.ndarray,
    gene_ids: np.ndarray,
    return_pt: bool = True,
    append_cls: bool = True,
    include_zero_gene: bool = False,
    cls_id: int = "<cls>",
    mod_type: np.ndarray = None,
    cls_id_mod_type: int = None,
) -> List[Tuple[Union[torch.Tensor, np.ndarray]]]:
    """
    Tokenize a batch of data. Returns a list of tuple (gene_id, count).

    Args:
        data (array-like): A batch of data, with shape (batch_size, n_features).
            n_features equals the number of all genes.
        gene_ids (array-like): A batch of gene ids, with shape (n_features,).
        return_pt (bool): Whether to return torch tensors of gene_ids and counts,
            default to True.

    Returns:
        list: A list of tuple (gene_id, count) of non zero gene expressions.
    """
    if data.shape[1] != len(gene_ids):
        raise ValueError(
            f"Number of features in data ({data.shape[1]}) does not match "
            f"number of gene_ids ({len(gene_ids)})."
        )
    if mod_type is not None and data.shape[1] != len(mod_type):
        raise ValueError(
            f"Number of features in data ({data.shape[1]}) does not match "
            f"number of mod_type ({len(mod_type)})."
        )

    tokenized_data = []
    for i in range(len(data)):
        row = data[i]
        mod_types = None
        if include_zero_gene:
            values = row
            genes = gene_ids
            if mod_type is not None:
                mod_types = mod_type
        else:
            idx = np.nonzero(row)[0]
            values = row[idx]
            genes = gene_ids[idx]
            if mod_type is not None:
                mod_types = mod_type[idx]
        if append_cls:
            genes = np.insert(genes, 0, cls_id)
            values = np.insert(values, 0, 0)
            if mod_type is not None:
                mod_types = np.insert(mod_types, 0, cls_id_mod_type)
        if return_pt:
            genes = torch.from_numpy(genes).long()
            values = torch.from_numpy(values).float()
            if mod_type is not None:
                mod_types = torch.from_numpy(mod_types).long()
        tokenized_data.append((genes, values, mod_types))
    return tokenized_data


# MODIFIED to accept and return indices
def pad_batch(
    batch: List[Tuple],
    max_len: int,
    vocab: Vocab,
    pad_token: str = "<pad>",
    pad_value: int = 0,
    cls_appended: bool = True,
    vocab_mod: Vocab = None,
    sample_indices: List[np.ndarray] = None,
) -> Tuple[Dict[str, torch.Tensor], List[np.ndarray]]:
    """
    Pad a batch of data.

    Args:
        batch (list): A list of tuple (gene_id, count).
        max_len (int): The maximum length of the batch.
        vocab (Vocab): The vocabulary containing the pad token.
        pad_token (str): The token to pad with.
        sample_indices (List[np.ndarray], optional): A list of pre-selected
            indices for each sample. If None, random sampling is performed
            for samples exceeding max_len. Defaults to None.

    Returns:
        Tuple[Dict[str, torch.Tensor], List[np.ndarray]]: A tuple containing:
            - A dictionary of padded tensors for 'genes' and 'values'.
            - A list of numpy arrays with the indices used for each sample.
    """
    max_ori_len = max(len(batch[i][0]) for i in range(len(batch)))
    max_len = min(max_ori_len, max_len)

    pad_id = vocab[pad_token]
    if vocab_mod is not None:
        mod_pad_id = vocab_mod[pad_token]
        
    gene_ids_list = []
    values_list = []
    mod_types_list = []
    used_indices_list = []

    for i in range(len(batch)):
        gene_ids, values, mod_types = batch[i]

        if len(gene_ids) > max_len:
            # If indices are provided for this sample, use them
            if sample_indices is not None and i < len(sample_indices):
                idx = sample_indices[i]
            # Otherwise, perform random sampling
            else:
                if not cls_appended:
                    idx = np.random.choice(len(gene_ids), max_len, replace=False)
                else:
                    # sample from non-CLS tokens and add CLS token back
                    idx = np.random.choice(len(gene_ids) - 1, max_len - 1, replace=False)
                    idx = idx + 1
                    idx = np.insert(idx, 0, 0)
            
            gene_ids = gene_ids[idx]
            values = values[idx]
            if mod_types is not None:
                mod_types = mod_types[idx]
            used_indices_list.append(idx)
        else:
            # If no sampling was needed, all original indices were used
            used_indices_list.append(np.arange(len(gene_ids)))

        if len(gene_ids) < max_len:
            gene_ids = torch.cat(
                [
                    gene_ids,
                    torch.full(
                        (max_len - len(gene_ids),), pad_id, dtype=gene_ids.dtype
                    ),
                ]
            )
            values = torch.cat(
                [
                    values,
                    torch.full((max_len - len(values),), pad_value, dtype=values.dtype),
                ]
            )
            if mod_types is not None:
                mod_types = torch.cat(
                    [
                        mod_types,
                        torch.full(
                            (max_len - len(mod_types),),
                            mod_pad_id,
                            dtype=mod_types.dtype,
                        ),
                    ]
                )

        gene_ids_list.append(gene_ids)
        values_list.append(values)
        if mod_types is not None:
            mod_types_list.append(mod_types)

    batch_padded = {
        "genes": torch.stack(gene_ids_list, dim=0),
        "values": torch.stack(values_list, dim=0),
    }
    if mod_types is not None and mod_types_list:
        batch_padded["mod_types"] = torch.stack(mod_types_list, dim=0)
        
    return batch_padded, used_indices_list


# MODIFIED to accept and return indices
def tokenize_and_pad_batch(
    data: np.ndarray,
    gene_ids: np.ndarray,
    max_len: int,
    vocab: Vocab,
    pad_token: str,
    pad_value: int,
    append_cls: bool = True,
    include_zero_gene: bool = False,
    cls_token: str = "<cls>",
    return_pt: bool = True,
    mod_type: np.ndarray = None,
    vocab_mod: Vocab = None,
    sample_indices: List[np.ndarray] = None,
) -> Tuple[Dict[str, torch.Tensor], List[np.ndarray]]:
    """
    Tokenize and pad a batch of data.

    Args:
        (Same as original, with the addition of sample_indices)
        sample_indices (List[np.ndarray], optional): A list of pre-selected
            indices for each sample. If None, random sampling is performed.
            Defaults to None.

    Returns:
        Tuple[Dict[str, torch.Tensor], List[np.ndarray]]: A tuple containing:
            - The dictionary of padded data ('genes', 'values').
            - A list of the indices used for each sample.
    """
    cls_id = vocab[cls_token]
    cls_id_mod_type = None
    if mod_type is not None:
        cls_id_mod_type = vocab_mod[cls_token]
        
    tokenized_data = tokenize_batch(
        data,
        gene_ids,
        return_pt=return_pt,
        append_cls=append_cls,
        include_zero_gene=include_zero_gene,
        cls_id=cls_id,
        mod_type=mod_type,
        cls_id_mod_type=cls_id_mod_type,
    )

    batch_padded, used_indices = pad_batch(
        tokenized_data,
        max_len,
        vocab,
        pad_token,
        pad_value,
        cls_appended=append_cls,
        vocab_mod=vocab_mod,
        sample_indices=sample_indices,  # Pass the indices here
    )
    return batch_padded, used_indices

# Currently still the same as scGPT, but maybe customize in the future
def random_mask_value(
    values: Union[torch.Tensor, np.ndarray],
    mask_ratio: float = 0.15,
    mask_value: int = -1,
    pad_value: int = 0,
) -> torch.Tensor:
    """
    Randomly mask a batch of data.

    Args:
        values (array-like):
            A batch of tokenized data, with shape (batch_size, n_features).
        mask_ratio (float): The ratio of genes to mask, default to 0.15.
        mask_value (int): The value to mask with, default to -1.
        pad_value (int): The value of padding in the values, will be kept unchanged.

    Returns:
        torch.Tensor: A tensor of masked data.
    """
    if isinstance(values, torch.Tensor):
        # it is crutial to clone the tensor, otherwise it changes the original tensor
        values = values.clone().detach().numpy()
    else:
        values = values.copy()

    for i in range(len(values)):
        row = values[i]
        non_padding_idx = np.nonzero(row - pad_value)[0]
        n_mask = int(len(non_padding_idx) * mask_ratio)
        mask_idx = np.random.choice(non_padding_idx, n_mask, replace=False)
        row[mask_idx] = mask_value
    return torch.from_numpy(values).float()