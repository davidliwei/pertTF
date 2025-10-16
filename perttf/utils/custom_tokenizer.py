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
from numpy.random import default_rng
# from transformers.tokenization_utils import PreTrainedTokenizer
# from transformers import AutoTokenizer, BertTokenizer
import collections

class SimpleVocab:
    """
    A simple, dependency-free vocabulary class to replace torchtext.vocab.Vocab.
    It handles string-to-index (stoi) and index-to-string (itos) mappings.
    """
    def __init__(self, tokens: list = None, special_tokens: list = None, vocab_path:str = None):
        """
        Args:
            tokens (list): A list of tokens (e.g., gene names) to build the vocabulary from.
            special_tokens (list, optional): A list of special tokens like '<pad>' or '<cls>'.
                                              These will be added to the beginning of the vocabulary.
        """
        if vocab_path is not None:
            self.from_json(vocab_path, special_tokens)
        elif tokens is not None:
            self.special_tokens = special_tokens if special_tokens else []

            # Combine special tokens and unique regular tokens
            all_tokens = self.special_tokens + sorted(list(set(tokens)))
            # Create integer-to-string mapping
            self.itos = {i:t for i,t in enumerate(all_tokens)}
            
            # Create string-to-integer mapping
            self.stoi = {token: i for i, token in self.itos.items()}
        
            # Set a default index for out-of-vocabulary tokens
            self._default_index = -1 # Uninitialized
            if "<pad>" in self.stoi:
                self.set_default_index(self.stoi["<pad>"])
            elif "<unk>" in self.stoi:
                self.set_default_index(self.stoi["<unk>"])
        else:
            self.itos = {}
            self.stoi = {}
            self.special_tokens = None
            

    def __len__(self):
        """Returns the size of the vocabulary."""
        return len(self.itos)

    def __getitem__(self, token: str) -> int:
        """Allows dictionary-style lookup (e.g., vocab['<pad>'])."""
        return self.stoi.get(token, self._default_index)

    def __call__(self, tokens: list) -> list:
        """
        Converts a list or a NumPy array of tokens to their corresponding indices.
        This method is optimized for NumPy arrays.
        """
        if isinstance(tokens, list) and tokens and isinstance(tokens[0], str):
            return [self.stoi.get(token, self._default_index) for token in tokens]

        if isinstance(tokens, list) and tokens and isinstance(tokens[0], list):
            return [[self.stoi.get(token, self._default_index) for token in sublist] for sublist in tokens]


    def set_default_index(self, index: int):
        """Sets the index to return for out-of-vocabulary tokens."""
        if not 0 <= index < len(self.itos):
            raise ValueError("Default index must be within the vocabulary size.")
        self._default_index = index
        # Update the default factory for collections.defaultdict if you were to use it
        # self.stoi.default_factory = lambda: self._default_index
        
    def get_itos(self):
        """Returns the list of tokens in order of their index."""
        return self.itos
    
    def from_json(self, json_file: str, special_tokens: list = None):
        with open(json_file) as f:
            stoi = json.load(f)
        self.special_tokens = [s for s in special_tokens if s in stoi] if special_tokens else []
        self.stoi = stoi
        self.itos = {i:t for i,t in enumerate(stoi)}

        # Create sorted arrays for fast NumPy lookups
        if "<pad>" in self.stoi:
            self.set_default_index(self.stoi["<pad>"])
        elif "<unk>" in self.stoi:
            self.set_default_index(self.stoi["<unk>"])
        

    def append(self, token: str):
        assert token not in self.stoi and token and type(token) == str, 'cannot append token, please check if token is str, not empty and new'
        self.itos[len(self.itos)] = token
        self.stoi[token] = len(self.itos)

# Smarter sampling function to get more non-zero expression for each cell during sentence generation
def weighted_sample(val, 
                    max_size, 
                    rng = default_rng(), 
                    mode = 'simple', 
                    non_zero_proportion = 0.7, 
                    fixed_ratio = False, 
                    hvg_nonhvg = None, 
                    non_hvg_size = 1000):
    
    # for each cell, sample from the hvgs and non-hvgs seperately
    hvg_size = max_size-non_hvg_size
    if mode == 'hvg':
        assert hvg_nonhvg is not None, 'hvg indices not given to sampling'
        # hvg_ids is a tuple, first element is hvg indices, second element is non-hvg_indices
        if non_hvg_size > 0:
            non_hvgs = rng.choice(len(hvg_nonhvg[1]), size = non_hvg_size, replace = False)
            samp_non_hvg_inds = hvg_nonhvg[1][non_hvgs]
        else:
            samp_non_hvg_inds = np.array([])
        if hvg_size >= len(hvg_nonhvg[0]):
            samp_hvg_inds = hvg_nonhvg[0]
        else:
            hvgs = rng.choice(len(hvg_nonhvg[0]), size = hvg_size, replace = False)
            samp_hvg_inds = hvg_nonhvg[0][hvgs]
        return np.concatenate([samp_hvg_inds, samp_non_hvg_inds])

    # for each cell, sample the expressed and zero genes seperately
    elif mode == 'expressed':
        nzero = np.where(val!=0)[0]
        zero = np.where(val==0)[0]
        non_zero_size = min(round(max_size*non_zero_proportion), len(nzero))
        if fixed_ratio:
            zero_size = min(max_size - non_zero_size, round(non_zero_size*(1-non_zero_proportion)/non_zero_proportion))
        else:
            zero_size = max_size - non_zero_size 
        nz_inds= rng.choice(len(nzero),size = non_zero_size, replace = False)
        z_inds= rng.choice(len(zero),size = zero_size, replace = False)
        return np.concatenate([nzero[nz_inds], zero[z_inds]])
    
    # randomly sample max_size number of genes for a cell
    else:
        return rng.choice(len(val),size = max_size, replace = False)

# This function remains unchanged
def tokenize_batch(
    data: np.ndarray,
    gene_ids: np.ndarray,
    return_pt: bool = True,
    append_cls: bool = True,
    include_zero_gene: bool = False,
    cls_id: int = 1,
    cls_value: int = -3,
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
            values = np.insert(values, 0, cls_value)
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
    vocab: SimpleVocab,
    pad_token: str = "<pad>",
    pad_value: int = -2,
    cls_appended: bool = True,
    vocab_mod: SimpleVocab = None,
    sample_indices: List[np.ndarray] = None,
    rng: default_rng = None,
    sampling_mode: str = 'simple',
    nonzero_prop = 0.7,
    fix_nonzero_prop = False,
    hvg_inds = None,
    non_hvg_size = 0
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
    rng = default_rng() if rng is None else rng # much faster sampling of genes
    if sample_indices is not None:
        assert len(sample_indices) == len(batch), 'if sample indices are provided, number of samples in batch must match number of sample indices'
    max_ori_len = max(len(batch[i][0]) for i in range(len(batch)))
    max_len = min(max_ori_len, max_len)

    pad_id = vocab[pad_token]
    if vocab_mod is not None:
        mod_pad_id = vocab_mod[pad_token]
        
    gene_ids_list = []
    values_list = []
    mod_types_list = []
    used_indices_list = []
    new_batch = []
    for i in range(len(batch)):
        gene_ids, values, mod_types = batch[i]

        if len(gene_ids) > max_len:
            # If indices are provided for this sample, use them
            if sample_indices is not None and i < len(sample_indices):
                idx = sample_indices[i]
            # Otherwise, perform random sampling
            else:
                if not cls_appended:
                    idx = weighted_sample(values, max_len, rng=rng, mode = sampling_mode, 
                                            non_zero_proportion=nonzero_prop, fixed_ratio=fix_nonzero_prop, 
                                            hvg_nonhvg = hvg_inds, non_hvg_size = non_hvg_size)
                else:
                    # sample from non-CLS tokens and add CLS token back
                    idx = weighted_sample(values[1:], max_len - 1, rng=rng, mode = sampling_mode, 
                                           non_zero_proportion=nonzero_prop, fixed_ratio=fix_nonzero_prop, 
                                           hvg_nonhvg = hvg_inds, non_hvg_size = non_hvg_size)
                    idx = idx + 1
                    idx = np.insert(idx, 0, 0)
            
            gene_ids = gene_ids[idx]
            values = values[idx]
            if mod_types is not None:
                mod_types = mod_types[idx]
            new_batch.append((gene_ids, values, mod_types))
            used_indices_list.append(idx)
        else:
            # If no sampling was needed, all original indices were used
            used_indices_list.append(gene_ids)

    # reset max_len to minimize padding for batch
    max_ori_len = max(len(new_batch[i][0]) for i in range(len(new_batch)))
    max_len = min(max_ori_len, max_len)
    for i in range(len(new_batch)):
        gene_ids, values, mod_types = new_batch[i]
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
    vocab: SimpleVocab,
    include_zero_gene: bool = False,
    pad_token: str = "<pad>",
    pad_value: int = -2,
    append_cls: bool = True,
    cls_token: str = "<cls>",
    cls_value: int = -3,
    return_pt: bool = True,
    mod_type: np.ndarray = None,
    vocab_mod: SimpleVocab = None,
    sample_indices: List[np.ndarray] = None,
    sampling_mode: str = 'simple',
    nonzero_prop: float = 0.7,
    fix_nonzero_prop: bool = False,
    hvg_inds = None,
    non_hvg_size = 1000
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
    if hvg_inds is not None:
        assert len(hvg_inds[0])+len(hvg_inds[1]) == len(gene_ids)
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
        cls_value=cls_value,
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
        sample_indices=sample_indices, # Pass the indices here
        sampling_mode = sampling_mode,
        nonzero_prop = nonzero_prop,
        fix_nonzero_prop = fix_nonzero_prop,
        hvg_inds = hvg_inds,
        non_hvg_size = non_hvg_size # if this is set to True, the data may contain padding
    )
    return batch_padded, used_indices

# Currently still the same as scGPT, but maybe customize in the future
def random_mask_value(
    values: Union[torch.Tensor, np.ndarray],
    mask_ratio: float = 0.15,
    mask_value: int = -1,
    pad_value: int = -2,
    cls_value: int = -3,
    rng: default_rng = None
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
    rng = default_rng() or rng # much faster sampling without replacement
    if isinstance(values, torch.Tensor):
        # it is crutial to clone the tensor, otherwise it changes the original tensor
        values = values.clone().detach().numpy()
    else:
        values = values.copy()

    for i in range(len(values)):
        row = values[i]
        non_padding_idx = np.intersect1d(np.nonzero(row - pad_value), np.nonzero(row - cls_value)) # only mask non padding or cls positions
        n_mask = int(len(non_padding_idx) * mask_ratio)
        mask_idx = rng.choice(non_padding_idx, n_mask, replace=False)
        row[mask_idx] = mask_value
    return torch.from_numpy(values).float()