import torch
from collections import OrderedDict
from perttf.utils.custom_tokenizer import SimpleVocab
from perttf.model.pertTF import PerturbationTFModel
import json
import os
from .logger import create_logger
# Try to take pretrained scGPT model files as input and output a vocab, config and pertTF model that has the weights loaded

def load_scGPT(folder):
    scGPT_items = os.listdir(folder)
    assert 'best_model.pt' in scGPT_items and 'vocab.json' in scGPT_items and 'args.json' in scGPT_items, 'scGPT folder is incomplete, please redownload must have at least a vocab file and model.pt'
    for f in scGPT_items:
        if f == 'best_model.pt':
            pretrained_dict = torch.load(folder+'/'+f)
        elif f == 'vocab.json':
            vocab = SimpleVocab(vocab_path=folder+'/'+f)
        elif f == 'args.json':
            with open(folder+'/'+f) as a:
                args = json.load(a)
    return {'pretrained_state':pretrained_dict, 
            'vocab':vocab, 
            'args':args}


def pertTF_from_scGPT_pretrain(config,
                               scGPT_dict, 
                               n_cls = 1,
                               n_perturb = 1,
                               num_batch_types = 1,
                               model = None, 
                               device = None, 
                               logger = create_logger()):
    """
    Transfers weights from scGPT pretrained model to pertTF model, handling both key name
    and shape mismatches automatically.
    """
    pretrained_dict, vocab, args = scGPT_dict['pretrained_state'], scGPT_dict['vocab'], scGPT_dict['args']
    transfer_config = {}
    transfer_config['nlayers'] = args['nlayers']
    transfer_config['nheads'] = args['nheads']
    transfer_config['layer_size'] = args['d_hid']
    transfer_config['embsize'] = args['embsize']
    n_layers_cls = args['n_layers_cls']
    config.update(transfer_config, allow_val_change=True)
    ntokens = len(vocab)
    if model is None:
        model = PerturbationTFModel(
        n_perturb, #n_perturb, # number of perturbation labels
        4, # layers
        1,
        ntokens,
        config.embsize, #embsize,
        config.nhead, #nhead,
        config.layer_size, #d_hid,
        config.nlayers, #nlayers,
        vocab=vocab,
        dropout=config.dropout,
        pad_token=config.pad_token,
        pad_value=config.pad_value,
        do_mvc=config.GEPC,
        do_dab=True if config.dab_weight >0 else False,
        use_batch_labels=config.use_batch_label,
        num_batch_labels= num_batch_types,#num_batch_types,
        domain_spec_batchnorm=config.DSBN,
        n_input_bins=config.n_bins,
        ecs_threshold=config.ecs_thres,
        explicit_zero_prob=config.explicit_zero_prob,
        use_fast_transformer=config.fast_transformer,
        pre_norm=config.pre_norm,
        n_cls=n_cls, # n_cls, # number of cell type labels
        nlayers_cls=n_layers_cls
        )
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # Create a new state dict to hold the renamed keys
    mapped_dict = OrderedDict()

    print("Mapping pretrained keys to target model keys...")
    for old_key, value in pretrained_dict.items():
        new_key = old_key
        
        # Map the QKV projection layer
        if config.fast_transformer:
            qkv_proj_w = 'qkv_proj.weight'
            qkv_proj_b = "qkv_proj.bias"
            out_proj = "out_proj"
        else:
            qkv_proj_w = "self_attn.in_proj_weight"
            qkv_proj_b = "self_attn.in_proj_bias"
            out_proj = "self_attn.out_proj"

        if "self_attn.Wqkv.weight" in old_key:
            new_key = old_key.replace("self_attn.Wqkv.weight", qkv_proj_w)
            logger.info(f"  - Mapping '{old_key}' -> '{new_key}'")
        elif "self_attn.in_proj_weight" in old_key:
            new_key = old_key.replace("self_attn.in_proj_weight", qkv_proj_w)
            logger.info(f"  - Mapping '{old_key}' -> '{new_key}'")
        elif "self_attn.Wqkv.bias" in old_key:
            new_key = old_key.replace("self_attn.Wqkv.bias", qkv_proj_b)
            logger.info(f"  - Mapping '{old_key}' -> '{new_key}'")
        elif "self_attn.in_proj_bias" in old_key:
            new_key = old_key.replace("self_attn.in_proj_bias", qkv_proj_b)
            logger.info(f"  - Mapping '{old_key}' -> '{new_key}'")   
        #  Map the output projection layer
        elif "self_attn.out_proj" in old_key:
            new_key = old_key.replace("self_attn.out_proj", out_proj)
            logger.info(f"  - Mapping '{old_key}' -> '{new_key}'")
        
        mapped_dict[new_key] = value

    # Get the state dict of new model
    model_dict = model.state_dict()
    
    # Filter the mapped dict for keys that exist in your model AND have matching shapes
    transfer_dict = {}
    transferred_keys = []
    skipped_keys = []

    for key, value in mapped_dict.items():
        if key in model_dict:
            if value.shape == model_dict[key].shape:
                transfer_dict[key] = value
                transferred_keys.append(key)
            else:
                # This will catch linear1 and linear2
                skipped_keys.append(f"{key} (shape mismatch: pretrained is {value.shape}, model is {model_dict[key].shape})")
        else:
            # This will catch layers that don't exist after renaming (e.g., qkv_proj.bias)
            skipped_keys.append(f"{key} (not found in new model)")

    # Overwrite the matching entries in your model's state dict
    model_dict.update(transfer_dict)
    
    # Load the updated state dict
    model.load_state_dict(model_dict)
    
    logger.info(f"\n✅ Transferred {len(transferred_keys)} of {len(pretrained_dict)} layers.")
    if skipped_keys:
        logger.info("🔍 Skipped layers:")
        for key in sorted(skipped_keys): # Sorted for cleaner output
            logger.info(f"  - {key}")
    return vocab, config, model, transferred_keys
