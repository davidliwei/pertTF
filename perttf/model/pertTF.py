from torch import nn, Tensor
from typing import Dict, Mapping, Optional, Tuple, Any, Union
from tqdm import trange

import numpy as np

import torch
from torch import nn
from torch.distributions import Bernoulli
import torch.nn.functional as F

import torch.distributed as dist

from .base_model import BaseModel

from torch.nn import TransformerEncoder, TransformerEncoderLayer
from .expr_sampler import DistributionGenerator
from .modules import (
    PertExpEncoder,
    PerturbationDecoder,
    PSDecoder,
    Batch2LabelEncoder,
    PertLabelEncoder
    )


class PerturbationTFModel(BaseModel):
    def __init__(self,
                 n_pert: int,
                 nlayers_pert: int,
                 n_ps: int,
                 *args, **kwargs):
        self.pred_lochness_next = kwargs.pop("pred_lochness_next", False) # additional optional parameter to ask whether to predict lochness scores
        ps_decoder2_nlayer = kwargs.pop("ps_decoder2_nlayer",3) # additional parameter to specify ps_decoder2 nlayer
        self.pert_pad_id = kwargs.pop("pert_pad_id", None) # get the pert_pad_id
        self.pert_dim = kwargs.pop('pert_dim', None)
        super().__init__(*args, **kwargs)
        # add perturbation encoder
        # variables are defined in super class
        d_model = self.d_model 
        pert_dim = d_model if self.pert_dim is None else self.pert_dim
        #self.pert_encoder = nn.Embedding(3, d_model, padding_idx=pert_pad_id)
        self.pert_encoder = PertLabelEncoder(n_pert, pert_dim, padding_idx=self.pert_pad_id)
        self.pert_exp_encoder = PertExpEncoder(d_model, self.pert_dim) 
        # the following is the perturbation decoder
        #n_pert = kwargs.get("n_perturb", 1) 
        #nlayers_pert = kwargs.get("nlayers_perturb", 3) 
        self.pert_decoder = PerturbationDecoder(d_model, n_pert, nlayers=nlayers_pert)
        # added: batch2 encoder, especially to model different cellular systems like cell line vs primary cells
        self.batch2_pad_id = None #kwargs.get("batch2_pad_id") if "batch2_pad_id" in kwargs else 2
        #self.batch2_encoder = nn.Embedding(2, d_model, padding_idx=self.batch2_pad_id)
        self.batch2_encoder = Batch2LabelEncoder(2, d_model) # should replace 2 to n_batch later
        self.n_pert = n_pert
        self.n_cls = kwargs.get("n_cls", 1) 
        
        
        if self.use_fast_transformer:
            nlayers = self.nlayers
            d_hid = self.d_hid
            nhead = self.nhead
            if self.fast_transformer_backend == 'flash':
                try:
                    from .modules import FlashTransformerEncoderLayerVarlen
                    
                    encoder_layers = FlashTransformerEncoderLayerVarlen(
                        d_model,
                        kwargs.get('nhead', nhead),
                        kwargs.get('d_hid', d_hid),
                        self.dropout,
                        batch_first=True,
                        norm_scheme=self.norm_scheme
                    )
                except Exception as e:
                    print(e)
                    print('DAO flash attention setup failed')
                    self.fast_transformer_backend == 'sdpa'

            if self.fast_transformer_backend == 'sdpa':
                print('trying pytorch SDPA')
                try:
                    from .modules import SDPATransformerEncoderLayer
                    encoder_layers = SDPATransformerEncoderLayer(
                        d_model,
                        nhead,
                        d_hid,
                        self.dropout,
                        batch_first=True,
                        norm_scheme=self.norm_scheme,
                    )
                except Exception as ee: 
                    print(ee)
                    print('pytorch sdpa attention setup failed, falling back to native pytorch attention')
                    self.use_fast_transformer = False
                    self.fast_transformer_backend == 'vanilla'
                    encoder_layers = TransformerEncoderLayer(
                        d_model, nhead, d_hid, self.dropout, batch_first=True
                    )
            self.transformer_encoder = TransformerEncoder(encoder_layers,  nlayers, enable_nested_tensor=False)

        
        # added: adding PS score decoder
        #self.n_ps = kwargs.get("n_ps") if "n_ps" in kwargs else 0
        self.n_ps = n_ps
        if self.n_ps > 0:
            self.ps_decoder = PSDecoder(d_model, self.n_ps, nlayers = nlayers_pert)
        else:
            self.ps_decoder = None
        if self.pred_lochness_next:
            self.ps_decoder2 = PSDecoder(d_model, 1, nlayers = ps_decoder2_nlayer, geneinput = True)
        else:
            self.ps_decoder2 = None

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
        sf: Optional[Tensor] = None,
        sf_next: Optional[Tensor] = None,
        CLS: bool = False,
        CCE: bool = False,
        MVC: bool = False,
        ECS: bool = False,
        PERTPRED: bool = False,
        do_sample: bool = False,
        PSPRED: bool = False,
        mvc_src: Tensor = None 
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

        if pert_labels is not None :
            pert_emb = self.pert_encoder(pert_labels)
            # transformmer output concatenate ?
            # note only input pert_labels should be concatenated, not pert_label_next
            #import pdb; pdb.set_trace()
            #tf_o_concat=torch.cat(
            #    [
            #        transformer_output_0,
            #        pert_emb.unsqueeze(1).repeat(1, transformer_output_0.shape[1], 1),
            #   ],
            #    dim=2,
            #)
            #transformer_output=self.pert_exp_encoder(tf_o_concat)
        else:
            #tf_o_concat = None # a placeholder
            pert_emb = None
        
        transformer_output=transformer_output_0
            
        output = {}
        output["contrastive_dict"] = {}
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
        output["contrastive_dict"]['orig_emb0'] = cell_emb_orig
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
            output["contrastive_dict"]['next_emb0'] = cell_emb_next
        else:
            tf_concat = None # add a placeholder
            cell_emb_next=cell_emb_orig
        
        cell_emb = cell_emb_orig
        output["cell_emb"] = cell_emb
        output["cell_emb_next"] = cell_emb_next

        if CLS:
            output["cls_output"] = self.cls_decoder(cell_emb)  # (batch, n_cls)
            output["cls_output_next"] = self.cls_decoder(cell_emb_next)  # (batch, n_cls)

        cur_gene_token_embs = self.encoder(mvc_src) if mvc_src is not None else self.cur_gene_token_embs
        if MVC and hasattr(self, 'mvc_decoder'):
            mvc_output = self.mvc_decoder(
                cell_emb
                if not self.use_batch_labels
                else torch.cat([cell_emb, batch_emb], dim=1),
                # else cell_emb + batch_emb,
                cur_gene_token_embs,
                target_size_factor = sf if self.distribution is not None else None
            )
            mvc_output_next = self.mvc_decoder(
                cell_emb_next
                if not self.use_batch_labels
                else torch.cat([cell_emb_next, batch_emb], dim=1),
                # else cell_emb + batch_emb,
                cur_gene_token_embs, # is it working well??
                target_size_factor = sf_next if self.distribution is not None else None
            )
            if self.explicit_zero_prob and False: # bernoulli sampling is meaningless here
                bernoulli = Bernoulli(probs=mvc_output["zero_probs"])
                output["mvc_output"] = bernoulli.sample() * mvc_output["pred"]

                bernoulli_n = Bernoulli(probs=mvc_output_next["zero_probs"])
                output["mvc_output_next"] = bernoulli.sample() * mvc_output_next["pred"]
            else:
                output["mvc_output"] = mvc_output  # (batch, seq_len)
                output["mvc_output_next"] = mvc_output_next  # (batch, seq_len)

        if ECS and hasattr(self, 'sim'):
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
            if self.pred_lochness_next:
                tf_concat=torch.cat([cell_emb_orig, pert_emb_next],dim=1)
                output["ps_output_next"] = self.ps_decoder2(tf_concat)  # this is the concatenation of cell embedding and predictive label (next)
            else:
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
        sf: Optional[Tensor] = None,
        output_to_cpu: bool = True,
        time_step: Optional[int] = None,
        return_np: bool = False,
        predict_expr = False,
        mvc_src: Tensor = None, # optional MVC tensor of gene ids for MVC decoder
        sample: bool = False, 
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

        if self.pred_lochness_next:
            shape_ps_next =  (N, 1) 
        else:
            shape_ps_next = shape_ps
        ps_outputs_next = array_func(shape_ps_next, dtype=float32_)
        
        expr_dict = {}
        if predict_expr:
            mlm_expr_shape = (N, src.size(1)) 
            mvc_expr_shape = (N, src.size(1)) if mvc_src is None else (N, mvc_src.size(1))
            mlm_outputs, mlm_zero_outputs = array_func(mlm_expr_shape, dtype=float32_), array_func(mlm_expr_shape, dtype=float32_)
            mvc_outputs, mvc_zero_outputs = array_func(mvc_expr_shape, dtype=float32_), array_func(mvc_expr_shape, dtype=float32_)
            mvc_next_outputs, mvc_next_zero_outputs = array_func(mvc_expr_shape, dtype=float32_), array_func(mvc_expr_shape, dtype=float32_)
            if self.distribution in ['nb', 'hnb', 'zinb', 'zig']:
                mvc_param2_outputs, mvc_next_param2_outputs = array_func(mvc_expr_shape, dtype=float32_), array_func(mvc_expr_shape, dtype=float32_)
        for i in trange(0, N, batch_size):
            src_d = src[i : i + batch_size].to(device)
            values_d = values[i : i + batch_size].to(device)
            src_key_padding_mask_d = src_key_padding_mask[i : i + batch_size].to(device)
            batch_labels_d = batch_labels[i : i + batch_size].to(device) if batch_labels is not None else None
            pert_labels_d = pert_labels[i : i + batch_size].to(device) if pert_labels is not None else None
            pert_labels_next_d = pert_labels_next[i : i + batch_size].to(device) if pert_labels_next is not None else None
            mvc_src_d = mvc_src[i:i+batch_size].to(device) if mvc_src is not None else None
            sf_d = sf[i:i+batch_size].to(device) if sf is not None else None
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
            tf_concat = None
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
            if self.pred_lochness_next:
                if self.ps_decoder2 is not None:
                    #import pdb; pdb.set_trace()
                    ps_output_next = self.ps_decoder2(tf_concat)
                if output_to_cpu:
                    ps_output_next = ps_output_next.cpu()
                if return_np:
                    ps_output_next = ps_output_next.numpy()
                ps_outputs_next[i : i + batch_size] = ps_output_next   
            else:
                ps_outputs_next[i : i + batch_size] = ps_outputs[i : i + batch_size]
            
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
                cur_gene_token_embs = self.encoder(mvc_src_d) if mvc_src_d is not None else self.cur_gene_token_embs
                mvc_output = self.mvc_decoder(
                                cell_emb if not self.use_batch_labels
                                else torch.cat([cell_emb, batch_emb], 
                                dim=1
                                ), # else cell_emb + batch_emb,
                                cur_gene_token_embs,
                                target_size_factor=sf_d,)
                if pert_labels_next_d is not None:
                    mvc_output_next = self.mvc_decoder(
                                cell_emb_next if not self.use_batch_labels
                                else torch.cat([cell_emb_next, batch_emb], 
                                dim=1
                                ), # else cell_emb + batch_emb,
                                cur_gene_token_embs,
                                target_size_factor=sf_d,)
                else:
                    mvc_output_next = mvc_output

                mlm_pred, mlm_zero_probs = mlm_output['pred'], mlm_output['zero_probs'] if self.explicit_zero_prob else torch.ones_like(mlm_output['pred'])
                if return_np:
                    mlm_pred, mlm_zero_probs = mlm_pred.cpu().numpy(), mlm_zero_probs.cpu().numpy() 
                mvc_generator = DistributionGenerator(self.distribution)
                mvc_output = mvc_generator.generate(mvc_output, sample = sample, to_numpy = return_np)
                mvc_output_next = mvc_generator.generate(mvc_output_next, sample = sample, to_numpy = return_np)
                mvc_pred, mvc_param2, mvc_zero_probs = mvc_output['pred'], mvc_output['param2'], mvc_output['zero_probs']
                mvc_pred_next, mvc_param2_next, mvc_zero_probs_next = mvc_output_next['pred'], mvc_output_next['param2'], mvc_output_next['zero_probs']
                mlm_outputs[i:i+batch_size], mvc_outputs[i:i+batch_size], mvc_next_outputs[i:i+batch_size] = mlm_pred, mvc_pred, mvc_pred_next
                if mvc_param2 is not None and mvc_param2_next is not None:
                    mvc_param2_outputs[i:i+batch_size], mvc_next_param2_outputs[i:i+batch_size]  = mvc_param2, mvc_param2_next
                if self.explicit_zero_prob:
                    mlm_zero_outputs[i:i+batch_size], mvc_zero_outputs[i:i+batch_size], mvc_next_zero_outputs[i:i+batch_size] = mlm_zero_probs, mvc_zero_probs,  mvc_zero_probs_next
                

        if predict_expr:
            expr_dict['mlm_expr'], expr_dict['mvc_expr'], expr_dict['mvc_next_expr'] = (mlm_outputs[:,1:], mvc_outputs[:,1:], mvc_next_outputs[:,1:])
            if self.explicit_zero_prob:
                expr_dict['mlm_expr_zero'], expr_dict['mvc_expr_zero'], expr_dict['mvc_next_expr_zero'] = (mlm_zero_outputs[:,1:], mvc_zero_outputs[:,1:], mvc_next_zero_outputs[:,1:])
            if self.distribution in ['nb', 'hnb', 'zinb', 'zig']:
                expr_dict['mvc_param2'], expr_dict['mvc_next_param2'] = (mvc_param2_outputs[:, 1:], mvc_next_param2_outputs[:, 1:])
        return outputs, outputs_next, pert_outputs, cls_outputs, ps_outputs, ps_outputs_next, expr_dict

