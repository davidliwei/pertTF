from typing import Dict, Mapping, Optional, Tuple, Any, Union
import torch
from torch import nn, Tensor
from torch.distributions import Bernoulli
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.functional import scaled_dot_product_attention
from torch.nn.attention import SDPBackend, sdpa_kernel
# Try to import the MHA module from flash-attn v2
FLASH_ATTENTION_VERSION = None
flash_attn_qkvpacked_func = None
flash_attn_varlen_func = None

# 2. Try to import the newest version first
try:
    # Assuming 'flash_attn_interface' is the newer package/module
    from flash_attn_interface import flash_attn_qkvpacked_func, flash_attn_varlen_func
    FLASH_ATTENTION_VERSION = '3'
    print("✅ Detected Flash Attention v3.")
except ImportError:
    # 3. If the first import fails, try the next one
    try:
        from flash_attn import flash_attn_qkvpacked_func, flash_attn_varlen_func
        FLASH_ATTENTION_VERSION = '2'
        print("✅ Detected Flash Attention v2.")
    except ImportError:
        # 4. If all imports fail, provide a notice
        print("⚠️ Flash Attention not installed. Model will use standard attention.")


class Empty:
    pass

class FlashTransformerEncoderLayerVarlen(nn.Module):
    """
    Alternative implementation that uses flash_attn_varlen_func for better handling
    of sequences with different lengths (padding).
    """
    
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        layer_norm_eps=1e-5,
        batch_first=True,
        device=None,
        dtype=None,
        norm_scheme="post",  # "pre" or "post"
        causal=False,
        bias = True,
        use_flash_attn = True
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.flash_version = FLASH_ATTENTION_VERSION
        self.d_model = d_model
        self.nhead = nhead
        self.batch_first = batch_first
        self.causal = causal
        
        # Multi-head attention components
        self.head_dim = d_model // nhead
        assert self.head_dim * nhead == d_model, f"d_model ({d_model}) must be divisible by nhead ({nhead})"
        self.self_attn = Empty() # Dummy code because TransformerEncoder expects self.self_attn.batch_first
        self.self_attn.batch_first = batch_first
        # Linear projections for Q, K, V
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=bias, **factory_kwargs)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias, **factory_kwargs)
        
        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        
        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = self._get_activation_fn(activation)
        self.norm_scheme = norm_scheme
        if self.norm_scheme not in ["pre", "post"]:
            raise ValueError(f"norm_scheme should be pre or post, not {norm_scheme}")

    @staticmethod
    def _get_activation_fn(activation):
        if activation == "relu":
            return F.relu
        elif activation == "gelu":
            return F.gelu
        raise RuntimeError(f"activation should be relu/gelu, not {activation}")

    def _flash_attention(self, x, key_padding_mask=None):
        """
        Perform flash attention on the input tensor using variable length attention
        when padding mask is present.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            key_padding_mask: Boolean mask of shape (batch_size, seq_len)
                             True indicates positions to be masked
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        qkv = self.qkv_proj(x)  # (batch_size, seq_len, 3 * d_model)
        
        # Reshape to separate Q, K, V
        qkv = qkv.reshape(batch_size, seq_len, 3, self.nhead, self.head_dim)
        #print(qkv.shape)
        qkv = qkv.permute(2, 0, 1, 3, 4)  # (3, batch_size, seq_len, nhead, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (batch_size, seq_len, nhead, head_dim)
        #print(qkv.shape)
        # Check if we have any padding
        if key_padding_mask is not None:
            # Convert to boolean if needed
            if key_padding_mask.dtype != torch.bool:
                key_padding_mask = key_padding_mask.bool()
            
            # Check if there's actual padding
            if not key_padding_mask.any():
                key_padding_mask = None
        
        if key_padding_mask is None:
            # No padding mask - use the efficient packed version
            # Repack for flash_attn_qkvpacked_func
            qkv_for_flash = torch.stack([q, k, v], dim=4)  # (batch, seq_len, nhead, head_dim, 3)
            #print(qkv_for_flash.shape)
            qkv_for_flash = qkv_for_flash.permute(0, 1, 4, 2, 3)  # (batch, seq_len, 3, nheads, head_dim)
            #print(qkv_for_flash.shape)
            if self.flash_version == '3':
                attn_output = flash_attn_qkvpacked_func(
                    qkv_for_flash,
                    softmax_scale=None,
                    causal=self.causal,
                
                )
            elif self.flash_version == '2':
                attn_output = flash_attn_qkvpacked_func(
                    qkv_for_flash,
                    dropout_p=self.dropout1.p if self.training else 0.0,
                    softmax_scale=None,
                    causal=self.causal,
                    return_attn_probs=False,
                )
            #print(attn_output.shape)
            # attn_output is (batch, seq_len, nhead, head_dim)
        else:
            # Use variable length attention for sequences with padding
            # Calculate actual sequence lengths
            seqlens = (~key_padding_mask).sum(dim=1, dtype=torch.int32)
            
            # Create cumulative sequence lengths
            cu_seqlens = torch.cat([
                torch.tensor([0], dtype=torch.int32, device=x.device),
                seqlens.cumsum(dim=0, dtype=torch.int32)
            ])
            
            # Create mask for valid positions
            valid_mask = ~key_padding_mask  # True for valid positions
            
            # Pack sequences by removing padded positions
            q_packed_list = []
            k_packed_list = []
            v_packed_list = []
            
            for b in range(batch_size):
                valid_indices = valid_mask[b]
                if valid_indices.any():  # Only process if there are valid tokens
                    q_packed_list.append(q[b][valid_indices])
                    k_packed_list.append(k[b][valid_indices])
                    v_packed_list.append(v[b][valid_indices])
            
            # Handle edge case where all sequences might be fully padded
            if q_packed_list:
                q_packed = torch.cat(q_packed_list, dim=0)  # (total_valid_tokens, nhead, head_dim)
                k_packed = torch.cat(k_packed_list, dim=0)
                v_packed = torch.cat(v_packed_list, dim=0)
                
                # Apply variable length flash attention
                max_seqlen = int(seqlens.max().item())
                if self.flash_version == '3':
                    attn_output_packed = flash_attn_varlen_func(
                        q_packed,
                        k_packed, 
                        v_packed,
                        cu_seqlens_q=cu_seqlens,
                        cu_seqlens_k=cu_seqlens,
                        max_seqlen_q=max_seqlen,
                        max_seqlen_k=max_seqlen,
                        softmax_scale=None,
                        causal=self.causal,
                    
                    )
                elif self.flash_version == '2':
                    attn_output_packed = flash_attn_varlen_func(
                        q_packed,
                        k_packed, 
                        v_packed,
                        cu_seqlens_q=cu_seqlens,
                        cu_seqlens_k=cu_seqlens,
                        max_seqlen_q=max_seqlen,
                        max_seqlen_k=max_seqlen,
                        dropout_p=self.dropout1.p if self.training else 0.0,
                        softmax_scale=None,
                        causal=self.causal,
                        return_attn_probs=False,
                    )
                # attn_output_packed shape: (total_valid_tokens, nhead, head_dim)
                
                # Unpack the output back to original shape with padding
                attn_output = torch.zeros(
                    batch_size, seq_len, self.nhead, self.head_dim,
                    dtype=attn_output_packed.dtype,
                    device=attn_output_packed.device
                )
                
                # Scatter the results back to their original positions
                start_idx = 0
                for b in range(batch_size):
                    valid_indices = valid_mask[b]
                    if valid_indices.any():
                        num_valid = valid_indices.sum().item()
                        attn_output[b][valid_indices] = attn_output_packed[start_idx:start_idx + num_valid]
                        start_idx += num_valid
            else:
                # All sequences are fully padded
                attn_output = torch.zeros(
                    batch_size, seq_len, self.nhead, self.head_dim,
                    dtype=x.dtype,
                    device=x.device
                )
        
        # Reshape output: (batch, seq_len, nhead, head_dim) -> (batch, seq_len, d_model)
        attn_output = attn_output.reshape(batch_size, seq_len, self.d_model)
        
        # Output projection
        return self.out_proj(attn_output)

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
                Shape: (batch_size, seq_len, d_model) if batch_first=True
            src_mask: the mask for the src sequence (optional).
                Note: FlashAttention v2 has limited support for arbitrary attention masks
            src_key_padding_mask: the mask for the src keys per batch (optional).
                Shape: (batch_size, seq_len), True means ignore/mask that position
                Can be bool or float tensor (will be converted to bool)

        Returns:
            Tensor of shape (batch_size, seq_len, d_model)
        """
        
        if src_mask is not None:
            # FlashAttention v2 supports causal masks natively but arbitrary masks need special handling
            if not self.causal:
                raise ValueError(
                    "FlashAttention v2 only supports causal masks natively. "
                    "For arbitrary attention masks, consider using standard attention."
                )
        
        # Ensure batch_first format
        if not self.batch_first:
            src = src.transpose(0, 1)
        
        if self.norm_scheme == "pre":
            # Pre-normalization
            src = self.norm1(src)
            src2 = self._flash_attention(src, key_padding_mask=src_key_padding_mask)
            src = src + self.dropout1(src2)
            
            src = self.norm2(src)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)
        else:
            # Post-normalization
            src2 = self._flash_attention(src, key_padding_mask=src_key_padding_mask)
            src = src + self.dropout1(src2)
            src = self.norm1(src)
            
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)
            src = self.norm2(src)
        
        # Convert back if needed
        if not self.batch_first:
            src = src.transpose(0, 1)
            
        return src
    

# Try to use other backends for attention
class SDPATransformerEncoderLayer(nn.Module):
    """
    A Transformer Encoder Layer that uses torch.nn.functional.scaled_dot_product_attention
    to automatically leverage the best available attention backend (e.g., FlashAttention, cuDNN).
    """
    
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        layer_norm_eps=1e-5,
        batch_first=True,
        device=None,
        dtype=None,
        norm_scheme="post",
        causal=False,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        
        self.d_model = d_model
        self.nhead = nhead
        self.batch_first = batch_first
        self.causal = causal
        self.self_attn = Empty()
        self.self_attn.batch_first = batch_first
        self.head_dim = d_model // nhead
        assert self.head_dim * nhead == d_model, f"d_model ({d_model}) must be divisible by nhead ({nhead})"
        
        # REMOVED: The FlashSelfAttention module and use_flash_attn flag are no longer needed.
        
        # Linear projections for Q, K, V
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False, **factory_kwargs)
        self.out_proj = nn.Linear(d_model, d_model, bias=False, **factory_kwargs)
        
        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        # Layer normalization and dropouts
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = self._get_activation_fn(activation)
        self.norm_scheme = norm_scheme
        if self.norm_scheme not in ["pre", "post"]:
            raise ValueError(f"norm_scheme should be pre or post, not {norm_scheme}")

    @staticmethod
    def _get_activation_fn(activation):
        if activation == "relu":
            return F.relu
        elif activation == "gelu":
            return F.gelu
        raise RuntimeError(f"activation should be relu/gelu, not {activation}")

    # SIMPLIFIED: Replaced the complex _flash_attention method with a cleaner one.
    def _attention(self, x: Tensor, key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """
        Perform attention using PyTorch's scaled_dot_product_attention.
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        qkv = self.qkv_proj(x)  # (batch_size, seq_len, 3 * d_model)
        
        # Reshape and permute for SDPA
        qkv = qkv.reshape(batch_size, seq_len, 3, self.nhead, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, nhead, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (batch_size, nhead, seq_len, head_dim)
        
        attn_mask = key_padding_mask
        if attn_mask is not None:
            # attn_mask must be broadcastable to (batch, nhead, seq_len, seq_len)
            # We add the nhead and query_seq_len dimensions.
            attn_mask = attn_mask.view(batch_size, 1, 1, seq_len)


        # The entire logic for varlen and packed attention is replaced by this single call.
        # SDPA handles the padding mask and causality internally.
        with sdpa_kernel(SDPBackend.MATH):
            attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,  # Pass the reshaped mask here
            dropout_p=self.dropout1.p if self.training else 0.0,
            is_causal=self.causal
        )
        
        # Reshape output back to (batch_size, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.out_proj(attn_output)

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        if src_mask is not None and not self.causal:
            raise ValueError("SDPATransformerEncoderLayer only supports a causal mask via the 'causal' flag.")
        
        if not self.batch_first:
            src = src.transpose(0, 1)
        
        if self.norm_scheme == "pre":
            src_norm = self.norm1(src)
            attn_out = self._attention(src_norm, key_padding_mask=src_key_padding_mask)
            src = src + self.dropout1(attn_out)
            
            src_norm = self.norm2(src)
            ff_out = self.linear2(self.dropout(self.activation(self.linear1(src_norm))))
            src = src + self.dropout2(ff_out)
        else: # post-norm
            attn_out = self._attention(src, key_padding_mask=src_key_padding_mask)
            src = src + self.dropout1(attn_out)
            src = self.norm1(src)
            
            ff_out = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(ff_out)
            src = self.norm2(src)
        
        if not self.batch_first:
            src = src.transpose(0, 1)
            
        return src


"""
All Modules other than the flash attention modules are not yet imported in pertTF.py
This script hopes to refactor the module classes and provide better control and customization 
"""

class PerturbationDecoder(nn.Module):
    """
    Decoder for perturbation label prediction.
    revised from scGPT.ClsDecoder
    """

    def __init__(
        self,
        d_model: int,
        n_pert: int,
        nlayers: int = 3,
        activation: callable = nn.ReLU,
    ):
        super().__init__()
        # module list
        self._decoder = nn.ModuleList()
        for i in range(nlayers - 1):
            self._decoder.append(nn.Linear(d_model, d_model))
            self._decoder.append(nn.LayerNorm(d_model)) # changed normalization to before activation
            self._decoder.append(activation())
            
        self.out_layer = nn.Linear(d_model, n_pert)
        print(self)
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, embsize]
        """
        for layer in self._decoder:
            x = layer(x)
        return self.out_layer(x)


class PSDecoder(nn.Module):
    """
    Decoder for ps score prediction.
    revised from scGPT.ClsDecoder
    """

    def __init__(
        self,
        d_model: int,
        n_pert: int,
        nlayers: int = 3,
        activation: callable = nn.ReLU,
    ):
        super().__init__()
        # module list
        self._decoder = nn.ModuleList()
        for i in range(nlayers - 1):
            self._decoder.append(nn.Linear(d_model, d_model))
            self._decoder.append(nn.LayerNorm(d_model))
            self._decoder.append(activation())
            
        self.out_layer = nn.Linear(d_model, n_pert)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, embsize]
        """
        for layer in self._decoder:
            x = layer(x)
        return self.out_layer(x)

class Batch2LabelEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)  # (batch, embsize)
        x = self.enc_norm(x)
        return x

class BatchLabelEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)  # (batch, embsize)
        x = self.enc_norm(x)
        return x


class PertLabelEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self.enc_norm = nn.LayerNorm(embedding_dim)
        print(self)
    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)  # (batch, embsize)
        x = self.enc_norm(x)
        return x

    
class PertExpEncoder(nn.Module):
    """
    Concatenating gene expression embeddings (from transformers) with perturbation embeddings (from scGPT's PertEncoder)
    """
    def __init__(
        self,
        d_model: int
    ):
        super().__init__()
        d_in = d_model * 2 
        #d_in = d_model
        self.fc = nn.Sequential(
            nn.Linear(d_in, d_model),
            #nn.Sigmoid(),
            #nn.ReLU(),
            nn.LeakyReLU(),
            nn.Linear(d_model, d_model),
            #nn.ReLU(),
            #nn.Sigmoid(),
            nn.LeakyReLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            #nn.Linear(d_model, d_model),
        )

        print(self)
    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """x is the output of the transformer concatenated with perturbation embedding, (batch, d_model*2)"""
        # pred_value = self.fc(x).squeeze(-1)  
        return self.fc(x) # (batch, d_model)
    


class GeneEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)  # (batch, seq_len, embsize)
        x = self.enc_norm(x)
        return x


class ContinuousValueEncoder(nn.Module):
    """
    Encode real number values to a vector using neural nets projection.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_value: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(1, d_model)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.max_value = max_value

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len]
        """
        # TODO: test using actual embedding layer if input is categorical
        # expand last dimension
        x = x.unsqueeze(-1)
        # clip x to [-inf, max_value]
        x = torch.clamp(x, max=self.max_value)
        x = self.activation(self.linear1(x))
        x = self.linear2(x)
        x = self.norm(x)
        return self.dropout(x)


class CategoryValueEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x.long()
        x = self.embedding(x)  # (batch, seq_len, embsize)
        x = self.enc_norm(x)
        return x


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp

# added here for potential customisations
class ExprDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        explicit_zero_prob: bool = False,
        use_batch_labels: bool = False,
    ):
        super().__init__()
        d_in = d_model * 2 if use_batch_labels else d_model
        self.fc = nn.Sequential(
            nn.Linear(d_in, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(),
            nn.Linear(4*d_model, 1),
            
        )
        self.pred_act = nn.ELU()
        self.explicit_zero_prob = explicit_zero_prob
        if explicit_zero_prob:
            self.zero_logit = nn.Sequential(
                nn.Linear(d_in, d_model),
                nn.LeakyReLU(),
                nn.Linear(d_model, d_model),
                nn.LeakyReLU(),
                nn.Linear(d_model, 1),
            
            )
        print(self)
    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """x is the output of the transformer, (batch, seq_len, d_model)"""
        pred_value = self.pred_act(self.fc(x).squeeze(-1))+1  # (batch, seq_len)

        if not self.explicit_zero_prob:
            return dict(pred=pred_value)
        zero_logits = self.zero_logit(x).squeeze(-1)  # (batch, seq_len)
        zero_probs = torch.sigmoid(zero_logits)
        return dict(pred=pred_value, zero_probs=zero_probs)
        # TODO: note that the return currently is only for training. Since decoder
        # is not used in the test setting for the integration task, the eval/inference
        # logic is not implemented yet. However, remember to implement it when
        # the decoder is used in any test setting. The inference logic will need
        # to sample from the bernoulli distribution with the zero_probs.

# added here for potential customisations
class ClsDecoder(nn.Module):
    """
    Decoder for classification task.
    """

    def __init__(
        self,
        d_model: int,
        n_cls: int,
        nlayers: int = 3,
        activation: callable = nn.ReLU,
    ):
        super().__init__()
        # module list
        self._decoder = nn.ModuleList()
        for i in range(nlayers - 1):
            self._decoder.append(nn.Linear(d_model, d_model))
            self._decoder.append(nn.LayerNorm(d_model)) # switched the normalization to before activation
            self._decoder.append(activation())
            
        self.out_layer = nn.Linear(d_model, n_cls)
        print(self)
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, embsize]
        """
        for layer in self._decoder:
            x = layer(x)
        return self.out_layer(x)

# added here for potential customisations
class MVCDecoder(nn.Module):
    """
    Decoder for the masked value prediction for cell embeddings.
    """

    def __init__(
        self,
        d_model: int,
        arch_style: str = "inner product",
        query_activation: nn.Module = nn.Sigmoid,
        hidden_activation: nn.Module = nn.PReLU,
        explicit_zero_prob: bool = False,
        use_batch_labels: bool = False,
    ) -> None:
        """
        Args:
            d_model (:obj:`int`): dimension of the gene embedding.
            arch_style (:obj:`str`): architecture style of the decoder, choice from
                1. "inner product" or 2. "concat query" or 3. "sum query".
            query_activation (:obj:`nn.Module`): activation function for the query
                vectors.
            hidden_activation (:obj:`nn.Module`): activation function for the hidden
                layers.
        """
        super().__init__()
        d_in = d_model * 2 if use_batch_labels else d_model
        if arch_style in ["inner product", "inner product, detach"]:
            self.gene2query = nn.Linear(d_model, d_model)
            self.query_activation = nn.LeakyReLU()#query_activation()
            self.W = nn.Linear(d_model, d_in, bias=False)
            if explicit_zero_prob:  # by default, gene-wise prob rate
                self.W_zero_logit = nn.Linear(d_model, d_in)
        elif arch_style == "concat query":
            self.gene2query = nn.Linear(d_model, 64)
            self.query_activation = query_activation()
            self.fc1 = nn.Linear(d_model + 64, 64)
            self.hidden_activation = hidden_activation()
            self.fc2 = nn.Linear(64, 1)
        elif arch_style == "sum query":
            self.gene2query = nn.Linear(d_model, d_model)
            self.query_activation = query_activation()
            self.fc1 = nn.Linear(d_model, 64)
            self.hidden_activation = hidden_activation()
            self.fc2 = nn.Linear(64, 1)
        else:
            raise ValueError(f"Unknown arch_style: {arch_style}")
        self.pred_act = nn.ELU()
        self.arch_style = arch_style
        self.do_detach = arch_style.endswith("detach")
        self.explicit_zero_prob = explicit_zero_prob
        print(self)
    def forward(
        self, cell_emb: Tensor, gene_embs: Tensor
    ) -> Union[Tensor, Dict[str, Tensor]]:
        """
        Args:
            cell_emb: Tensor, shape (batch, embsize=d_model)
            gene_embs: Tensor, shape (batch, seq_len, embsize=d_model)
        """
        gene_embs = gene_embs.detach() if self.do_detach else gene_embs
        if self.arch_style in ["inner product", "inner product, detach"]:
            query_vecs = self.query_activation(self.gene2query(gene_embs))
            cell_emb = cell_emb.unsqueeze(2)  # (batch, embsize, 1)
            # the pred gene expr values, # (batch, seq_len)
            pred_value = self.pred_act(torch.bmm(self.W(query_vecs), cell_emb).squeeze(2))
            if not self.explicit_zero_prob:
                return dict(pred=pred_value)
            # zero logits need to based on the cell_emb, because of input exprs
            zero_logits = torch.bmm(self.W_zero_logit(query_vecs), cell_emb).squeeze(2)
            zero_probs = torch.sigmoid(zero_logits)
            return dict(pred=pred_value, zero_probs=zero_probs)
        elif self.arch_style == "concat query":
            query_vecs = self.query_activation(self.gene2query(gene_embs))
            # expand cell_emb to (batch, seq_len, embsize)
            cell_emb = cell_emb.unsqueeze(1).expand(-1, gene_embs.shape[1], -1)

            h = self.hidden_activation(
                self.fc1(torch.cat([cell_emb, query_vecs], dim=2))
            )
            if self.explicit_zero_prob:
                raise NotImplementedError
            return self.pred_act(self.fc2(h).squeeze(2))  # (batch, seq_len)
        elif self.arch_style == "sum query":
            query_vecs = self.query_activation(self.gene2query(gene_embs))
            cell_emb = cell_emb.unsqueeze(1)

            h = self.hidden_activation(self.fc1(cell_emb + query_vecs))
            if self.explicit_zero_prob:
                raise NotImplementedError
            return self.pred_act(self.fc2(h).squeeze(2))  # (batch, seq_len)

