from torch import nn, Tensor
from typing import Dict, Mapping, Optional, Tuple, Any, Union
#from scgpt.model import BatchLabelEncoder
from tqdm import trange

import numpy as np

import torch
from torch import nn
from torch.distributions import Bernoulli
import torch.nn.functional as F

import torch.distributed as dist



import scgpt as scg
from scgpt.model import TransformerModel

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
            self._decoder.append(activation())
            self._decoder.append(nn.LayerNorm(d_model))
        self.out_layer = nn.Linear(d_model, n_pert)

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
            self._decoder.append(activation())
            self._decoder.append(nn.LayerNorm(d_model))
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
            nn.Sigmoid(),#nn.ReLU(),#nn.LeakyReLU(),
            nn.Linear(d_model, d_model),
            #nn.ReLU(),
            nn.Sigmoid(),
            nn.Linear(d_model, d_model),
            #nn.LayerNorm(d_model),
            #nn.Linear(d_model, d_model),
        )


    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """x is the output of the transformer concatenated with perturbation embedding, (batch, d_model*2)"""
        # pred_value = self.fc(x).squeeze(-1)  
        return self.fc(x) # (batch, d_model)



class PerturbationTFModel(TransformerModel):
    def __init__(self,
                 n_pert: int,
                 nlayers_pert: int,
                 n_ps: int,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        # add perturbation encoder
        # variables are defined in super class
        d_model = self.d_model
        self.pert_pad_id = kwargs.get("pert_pad_id") if "pert_pad_id" in kwargs else 2
        pert_pad_id = self.pert_pad_id
        #self.pert_encoder = nn.Embedding(3, d_model, padding_idx=pert_pad_id)
        self.pert_encoder = PertLabelEncoder(n_pert, d_model, padding_idx=pert_pad_id)

        self.pert_exp_encoder = PertExpEncoder (d_model) 

        # the following is the perturbation decoder
        #n_pert = kwargs.get("n_perturb") if "n_perturb" in kwargs else 1
        #nlayers_pert = kwargs.get("nlayers_perturb") if "nlayers_perturb" in kwargs else 3
        self.pert_decoder = PerturbationDecoder(d_model, n_pert, nlayers=nlayers_pert)

        # added: batch2 encoder, especially to model different cellular systems like cell line vs primary cells
        self.batch2_pad_id = None #kwargs.get("batch2_pad_id") if "batch2_pad_id" in kwargs else 2
        #self.batch2_encoder = nn.Embedding(2, d_model, padding_idx=self.batch2_pad_id)
        self.batch2_encoder = Batch2LabelEncoder(2, d_model) # should replace 2 to n_batch later
        self.n_pert = n_pert
        self.n_cls = kwargs.get("n_cls") if "n_cls" in kwargs else 1

        # added: adding PS score decoder
        #self.n_ps = kwargs.get("n_ps") if "n_ps" in kwargs else 0
        self.n_ps = n_ps
        if self.n_ps > 0:
            self.ps_decoder = PSDecoder(d_model, self.n_ps, nlayers = nlayers_pert)
        else:
            self.ps_decoder = None

    # rewrite encode function
    def _encode(
        self,
        src: Tensor,
        values: Tensor,
        src_key_padding_mask: Tensor,
        batch_labels: Optional[Tensor] = None,  # (batch,)
        input_pert_flags: Optional[Tensor] = None,
    ) -> Tensor:
        #print('_encode batch labels:')
        #print(batch_labels)
        self._check_batch_labels(batch_labels)

        src = self.encoder(src)  # (batch, seq_len, embsize)
        self.cur_gene_token_embs = src

        values = self.value_encoder(values)  # (batch, seq_len, embsize)

        if self.input_emb_style == "scaling":
            values = values.unsqueeze(2)
            total_embs = src * values
        else:
            total_embs = src + values

        # add additional perturbs
        if input_pert_flags is not None:
            perts = self.pert_encoder(input_pert_flags)  # (batch, seq_len, embsize)
            #import pdb; pdb.set_trace()
            perts_expand = perts.unsqueeze(1).repeat(1, total_embs.shape[1], 1)
            total_embs = total_embs + perts_expand

        # batch2 TODO: use batch_encoder instead
        if batch_labels is not None:
            batch2_embs = self.batch2_encoder(batch_labels)
            #import pdb; pdb.set_trace()
            batch2_embs = batch2_embs.unsqueeze(1).repeat(1, total_embs.shape[1], 1)
            total_embs = total_embs + batch2_embs

        # dsbn and batch normalization
        if getattr(self, "dsbn", None) is not None:
            batch_label = int(batch_labels[0].item())
            total_embs = self.dsbn(total_embs.permute(0, 2, 1), batch_label).permute(
                0, 2, 1
            )  # the batch norm always works on dim 1
        elif getattr(self, "bn", None) is not None:
            total_embs = self.bn(total_embs.permute(0, 2, 1)).permute(0, 2, 1)


        output = self.transformer_encoder(
            total_embs, src_key_padding_mask=src_key_padding_mask
        )
        return output  # (batch, seq_len, embsize)

    def forward(
        self,
        src: Tensor,
        values: Tensor,
        src_key_padding_mask: Tensor,
        batch_labels: Optional[Tensor] = None,
        pert_labels: Optional[Tensor] = None, 
        pert_labels_next: Optional[Tensor] = None, 
        CLS: bool = False,
        CCE: bool = False,
        MVC: bool = False,
        ECS: bool = False,
        PERTPRED: bool = False,
        do_sample: bool = False,
        PSPRED: bool = False,
    ) -> Mapping[str, Tensor]:
        """
        Args:
            src (:obj:`Tensor`): token ids, shape [batch_size, seq_len]
            values (:obj:`Tensor`): token values, shape [batch_size, seq_len]
            src_key_padding_mask (:obj:`Tensor`): mask for src, shape [batch_size,
                seq_len]
            batch_labels (:obj:`Tensor`): batch labels, shape [batch_size]
            pert_labels (:obj:`Tensor`): perturbation labels, shape [batch_size]
            pert_labels_next (:obj:`Tensor`): perturbation labels for prediction, shape [batch_size]
            CLS (:obj:`bool`): if True, return the celltype classification objective
                (CLS) output
            CCE (:obj:`bool`): if True, return the contrastive cell embedding objective
                (CCE) output
            MVC (:obj:`bool`): if True, return the masked value prediction for cell
                embedding MVC output
            ECS (:obj:`bool`): if True, return the elastic cell similarity objective
                (ECS) output.
            PERTPRED (:obj:`bool`): if True, return the perturbation prediction
                (PERTPRED) output. 
            PSPRED (:obj:`bool`): if True, return the PS score prediction 
                (PERTPRED) output. 

        Returns:
            dict of output Tensors.
        """
        #print('forward batch labels:')
        #print(batch_labels)
        # call the super forward function
        #output = super().forward(
        #    src,
        #    values,
        #    src_key_padding_mask,
        #    batch_labels=batch_labels,
        #    CLS=CLS,
        #    CCE=CCE,
        #    MVC=MVC,
        #    ECS=ECS,
        #    do_sample=do_sample,
        #)

        # or, rewrite the forward function
        transformer_output_0 = self._encode(
            src, values, src_key_padding_mask, batch_labels,
            input_pert_flags= pert_labels, # Do we use pert_flags for transformer input?
        )
        if self.use_batch_labels:
            batch_emb = self.batch_encoder(batch_labels)  # (batch, embsize)

        pert_emb = None
        if pert_labels is not None :
            pert_emb = self.pert_encoder(pert_labels)
        # transformmer output concatenate ?
        # note only input pert_labels should be concatenated, not pert_label_next
        if pert_labels is not None and False:

            #import pdb; pdb.set_trace()
            tf_o_concat=torch.cat(
                [
                    transformer_output_0,
                    pert_emb.unsqueeze(1).repeat(1, transformer_output_0.shape[1], 1),
                ],
                dim=2,
            )
            transformer_output=self.pert_exp_encoder(tf_o_concat)
        else:
            transformer_output=transformer_output_0
            
        output = {}
        mlm_output = self.decoder(
            transformer_output
            if not self.use_batch_labels
            else torch.cat(
                [
                    transformer_output,
                    batch_emb.unsqueeze(1).repeat(1, transformer_output.shape[1], 1),
                ],
                dim=2,
            ),
            # else transformer_output + batch_emb.unsqueeze(1),
        )
        # zero_probs is actually non-zero probability for the Bernoulli
        if self.explicit_zero_prob and do_sample:
            bernoulli = Bernoulli(probs=mlm_output["zero_probs"])
            output["mlm_output"] = bernoulli.sample() * mlm_output["pred"]
        else:
            output["mlm_output"] = mlm_output["pred"]  # (batch, seq_len)
        if self.explicit_zero_prob:
            output["mlm_zero_probs"] = mlm_output["zero_probs"]

        cell_emb_orig = self._get_cell_emb_from_layer(transformer_output, values)        
        
        #  concatenate cell embedding with perturbation embedding to generate next cell embedding
        if pert_labels_next is not None: #and False:
            #import pdb; pdb.set_trace()
            pert_emb_next = self.pert_encoder(pert_labels_next)
            tf_concat=torch.cat(
                [
                    cell_emb_orig,
                    pert_emb_next,
                ],
                dim=1,
            )
            #tf_concat = cell_emb_orig + pert_emb_next
            cell_emb_next=self.pert_exp_encoder(tf_concat)
        else:
            cell_emb_next=cell_emb_orig
        
        cell_emb = cell_emb_orig
        output["cell_emb"] = cell_emb
        output["cell_emb_next"] = cell_emb_next

        if CLS:
            output["cls_output"] = self.cls_decoder(cell_emb)  # (batch, n_cls)
            output["cls_output_next"] = self.cls_decoder(cell_emb_next)  # (batch, n_cls)
        if CCE:
            cell1 = cell_emb
            transformer_output2 = self._encode(
                src, values, src_key_padding_mask, batch_labels
            )
            cell2 = self._get_cell_emb_from_layer(transformer_output2)

            # Gather embeddings from all devices if distributed training
            if dist.is_initialized() and self.training:
                cls1_list = [
                    torch.zeros_like(cell1) for _ in range(dist.get_world_size())
                ]
                cls2_list = [
                    torch.zeros_like(cell2) for _ in range(dist.get_world_size())
                ]
                dist.all_gather(tensor_list=cls1_list, tensor=cell1.contiguous())
                dist.all_gather(tensor_list=cls2_list, tensor=cell2.contiguous())

                # NOTE: all_gather results have no gradients, so replace the item
                # of the current rank with the original tensor to keep gradients.
                # See https://github.com/princeton-nlp/SimCSE/blob/main/simcse/models.py#L186
                cls1_list[dist.get_rank()] = cell1
                cls2_list[dist.get_rank()] = cell2

                cell1 = torch.cat(cls1_list, dim=0)
                cell2 = torch.cat(cls2_list, dim=0)
            # TODO: should detach the second run cls2? Can have a try
            cos_sim = self.sim(cell1.unsqueeze(1), cell2.unsqueeze(0))  # (batch, batch)
            labels = torch.arange(cos_sim.size(0)).long().to(cell1.device)
            output["loss_cce"] = self.creterion_cce(cos_sim, labels)
        if MVC:
            mvc_output = self.mvc_decoder(
                cell_emb
                if not self.use_batch_labels
                else torch.cat([cell_emb, batch_emb], dim=1),
                # else cell_emb + batch_emb,
                self.cur_gene_token_embs,
            )
            mvc_output_next = self.mvc_decoder(
                cell_emb_next
                if not self.use_batch_labels
                else torch.cat([cell_emb_next, batch_emb], dim=1),
                # else cell_emb + batch_emb,
                self.cur_gene_token_embs, # is it working well??
            )
            if self.explicit_zero_prob and do_sample:
                bernoulli = Bernoulli(probs=mvc_output["zero_probs"])
                output["mvc_output"] = bernoulli.sample() * mvc_output["pred"]

                bernoulli_n = Bernoulli(probs=mvc_output_next["zero_probs"])
                output["mvc_output_next"] = bernoulli.sample() * mvc_output_next["pred"]
            else:
                output["mvc_output"] = mvc_output["pred"]  # (batch, seq_len)
                output["mvc_output_next"] = mvc_output_next["pred"]  # (batch, seq_len)
            if self.explicit_zero_prob:
                output["mvc_zero_probs"] = mvc_output["zero_probs"]
                output["mvc_zero_probs_next"] = mvc_output_next["zero_probs"]
        if ECS:
            # Here using customized cosine similarity instead of F.cosine_similarity
            # to avoid the pytorch issue of similarity larger than 1.0, pytorch # 78064
            # normalize the embedding
            cell_emb_normed = F.normalize(cell_emb, p=2, dim=1)
            cos_sim = torch.mm(cell_emb_normed, cell_emb_normed.t())  # (batch, batch)

            # mask out diagnal elements
            mask = torch.eye(cos_sim.size(0)).bool().to(cos_sim.device)
            cos_sim = cos_sim.masked_fill(mask, 0.0)
            # only optimize positive similarities
            cos_sim = F.relu(cos_sim)

            output["loss_ecs"] = torch.mean(1 - (cos_sim - self.ecs_threshold) ** 2)

        if self.do_dab:
            output["dab_output"] = self.grad_reverse_discriminator(cell_emb)


        # get cell embedding
        if PERTPRED:
            #cell_emb = output["cell_emb"]
            output["pert_output"] = self.pert_decoder(cell_emb)  # (batch, n_cls)
            output["pert_output_next"] = self.pert_decoder(cell_emb_next)  # (batch, n_cls)

        # PS score prediction
        if PSPRED and self.ps_decoder is not None:
            output["ps_output"] = self.ps_decoder(cell_emb)
            output["ps_output_next"] = self.ps_decoder(cell_emb_next)  # (batch, n_cls)
        
        return output

    def encode_batch_with_perturb(
        self,
        src: Tensor,
        values: Tensor,
        src_key_padding_mask: Tensor,
        batch_size: int,
        batch_labels: Optional[Tensor] = None,
        pert_labels: Optional[Tensor] = None, # the first perturbation
        pert_labels_next: Optional[Tensor] = None, # the second perturbation
        output_to_cpu: bool = True,
        time_step: Optional[int] = None,
        return_np: bool = False,
        predict_expr = False
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        revised scgpt.TransformerModel.encode_batch but with additional perturbation
        prediction output
        Args:
            src (Tensor): shape [N, seq_len]
            values (Tensor): shape [N, seq_len]
            src_key_padding_mask (Tensor): shape [N, seq_len]
            batch_size (int): batch size for encoding
            batch_labels (Tensor): shape [N, n_batch_labels]
            output_to_cpu (bool): whether to move the output to cpu
            time_step (int): the time step index in the transformer output to return.
                The time step is along the second dimenstion. If None, return all.
            return_np (bool): whether to return numpy array

        Returns:
            output Tensor tuple of shape [N, seq_len, embsize] and [N, n_pert]
        """
        N = src.size(0)
        device = next(self.parameters()).device

        # initialize the output tensor
        array_func = np.zeros if return_np else torch.zeros
        float32_ = np.float32 if return_np else torch.float32
        shape = (
            (N, self.d_model)
            if time_step is not None
            else (N, src.size(1), self.d_model)
        )
        outputs = array_func(shape, dtype=float32_)
        outputs_next = array_func(shape, dtype=float32_)
        # added for perturbation predictions
        shape_perts = (N, self.n_pert) if time_step is not None else (N, src.size(1), self.n_pert)
        pert_outputs = array_func(shape_perts, dtype=float32_)
        
        # add for cls predictions
        shape_cls = (N, self.n_cls) if time_step is not None else (N, src.size(1), self.n_cls)
        cls_outputs = array_func(shape_cls, dtype=float32_)

        # added for PS score predictions
        shape_ps = (N, self.n_ps) if time_step is not None else (N, src.size(1), self.n_ps)
        ps_outputs =  array_func(shape_ps, dtype=float32_)
        
       
        
        expr_dict = {}
        if predict_expr:
            shape_expr = (N, src.size(1))
            mlm_outputs, mlm_zero_outputs = array_func(shape_expr, dtype=float32_), array_func(shape_expr, dtype=float32_)
            mvc_outputs, mvc_zero_outputs = array_func(shape_expr, dtype=float32_), array_func(shape_expr, dtype=float32_)
            mvc_next_outputs, mvc_next_zero_outputs = array_func(shape_expr, dtype=float32_), array_func(shape_expr, dtype=float32_)

        for i in trange(0, N, batch_size):
            src_d = src[i : i + batch_size].to(device)
            values_d = values[i : i + batch_size].to(device)
            src_key_padding_mask_d = src_key_padding_mask[i : i + batch_size].to(device)
            batch_labels_d = batch_labels[i : i + batch_size].to(device) if batch_labels is not None else None
            pert_labels_d = pert_labels[i : i + batch_size].to(device) if pert_labels is not None else None
            pert_labels_next_d = pert_labels_next[i : i + batch_size].to(device) if pert_labels_next is not None else None
            raw_output = self._encode(
                src_d,
                values_d,
                src_key_padding_mask_d,
                batch_labels_d,
                input_pert_flags= pert_labels_d, # Do we use pert_flags for transformer input?
            )
            output = raw_output.detach()
            if output_to_cpu:
                output = output.cpu()
            if return_np:
                output = output.numpy()
            if time_step is not None:
                output = output[:, time_step, :]
            outputs[i : i + batch_size] = output

            #import pdb; pdb.set_trace()
            cell_emb = self._get_cell_emb_from_layer(raw_output, values_d)
            
            if pert_labels_next_d is not None:
                pert_emb_next = self.pert_encoder(pert_labels_next_d)
                tf_concat=torch.cat(
                    [cell_emb,pert_emb_next], dim=1,
                )
                #tf_concat = cell_emb + pert_emb_next
                cell_emb_next=self.pert_exp_encoder(tf_concat)
                if output_to_cpu:
                    cell_emb_next_cpu = cell_emb_next.cpu()
                if return_np:
                    cell_emb_next_cpu = cell_emb_next_cpu.numpy()
                outputs_next[i : i + batch_size] = cell_emb_next_cpu
            else:
                #cell_emb_next=None
                outputs_next[i : i + batch_size] = output
            
            pert_output = self.pert_decoder(cell_emb)
            if output_to_cpu:
                pert_output = pert_output.cpu()
            if return_np:
                pert_output = pert_output.numpy()
            #if time_step is not None:
            #    pert_output = pert_output[:, time_step, :]
            pert_outputs[i : i + batch_size] = pert_output

            cls_output = self.cls_decoder(cell_emb)
            if output_to_cpu:
                cls_output = cls_output.cpu()
            if return_np:
                cls_output = cls_output.numpy()
            cls_outputs[i : i + batch_size] = cls_output

            # always check if ps decoder is used or not
            if self.ps_decoder is not None:
                ps_output = self.ps_decoder(cell_emb)
                if output_to_cpu:
                    ps_output = ps_output.cpu()
                if return_np:
                    ps_output = ps_output.numpy()
                ps_outputs[i : i + batch_size] = ps_output   
            
            
            if predict_expr:
                if self.use_batch_labels:
                    batch_emb = self.batch_encoder(batch_labels) 
                
                mlm_output = self.decoder(
                    raw_output
                    if not self.use_batch_labels
                    else torch.cat(
                    [
                       raw_output,
                       batch_emb.unsqueeze(1).repeat(1, raw_output.shape[1], 1),
                    ],
                    dim=2,
                ),
                # else transformer_output + batch_emb.unsqueeze(1),
                )
                mvc_output = self.mvc_decoder(
                                cell_emb if not self.use_batch_labels
                                else torch.cat([cell_emb, batch_emb], 
                                dim=1
                                ), # else cell_emb + batch_emb,
                            self.cur_gene_token_embs,)
                if pert_labels_next_d is not None:
                    mvc_output_next = self.mvc_decoder(
                                cell_emb_next if not self.use_batch_labels
                                else torch.cat([cell_emb_next, batch_emb], 
                                dim=1
                                ), # else cell_emb + batch_emb,
                            self.cur_gene_token_embs,)
                else:
                    mvc_output_next = mvc_output

                mlm_pred, mlm_zero_probs = mlm_output['pred'], mlm_output['zero_probs'] if self.explicit_zero_prob else 1
                mvc_pred, mvc_zero_probs = mvc_output['pred'], mvc_output['zero_probs'] if self.explicit_zero_prob else 1
                mvc_pred_next, mvc_zero_probs_next = mvc_output_next['pred'], mvc_output_next['zero_probs'] if self.explicit_zero_prob else 1
                if output_to_cpu:
                    mlm_pred, mlm_zero_probs = mlm_pred.cpu(), mlm_zero_probs.cpu() if self.explicit_zero_prob else 1
                    mvc_pred, mvc_zero_probs =  mvc_pred.cpu(), mvc_zero_probs.cpu() if self.explicit_zero_prob else 1
                    mvc_pred_next, mvc_zero_probs_next = mvc_pred_next.cpu(), mvc_zero_probs_next.cpu() if self.explicit_zero_prob else 1
                if return_np:
                    mlm_pred, mlm_zero_probs = mlm_pred.numpy(), mlm_zero_probs.numpy() if self.explicit_zero_prob else 1
                    mvc_pred, mvc_zero_probs =  mvc_pred.numpy(), mvc_zero_probs.numpy() if self.explicit_zero_prob else 1
                    mvc_pred_next, mvc_zero_probs_next = mvc_pred_next.numpy(), mvc_zero_probs_next.numpy() if self.explicit_zero_prob else 1

                mlm_outputs[i : i + batch_size], mlm_zero_outputs[i : i + batch_size] = mlm_pred, mlm_zero_probs
                mvc_outputs[i : i + batch_size], mvc_zero_outputs[i : i + batch_size] = mvc_pred, mvc_zero_probs
                mvc_next_outputs[i : i + batch_size], mvc_next_zero_outputs[i : i + batch_size] = mvc_pred_next, mvc_zero_probs_next


        if predict_expr:
            expr_dict['mlm_expr'] = (mlm_outputs, mlm_zero_outputs)
            expr_dict['mvc_expr'] = (mvc_outputs, mvc_zero_outputs)
            expr_dict['mvc_next_expr'] = (mvc_next_outputs, mvc_next_zero_outputs)

        return outputs, outputs_next, pert_outputs, cls_outputs, ps_outputs, expr_dict

