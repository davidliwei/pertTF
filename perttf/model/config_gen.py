import os
import sys
import wandb

from typing import Literal

from scgpt.utils import set_seed
import copy

# Check the Python interpreter being used
#print(sys.executable)

def generate_config(parameter_dict,
        project_name = "scGPT",
        wandb_mode : Literal["disabled","online","offline"] = "disabled"):
    # If it's not the desired interpreter, set the WANDB__EXECUTABLE environment variable
    # For example, if you want to use Python 3.8:
    os.environ["WANDB__EXECUTABLE"] = "/usr/local/bin/python"  # Replace with the actual path


    # settings for input and preprocessing

    parameter_dict['special_tokens'] = [parameter_dict.get('pad_token', '<pad>'), 
                                        parameter_dict.get('cls_token', '<cls>'), 
                                        "<eoc>"]

    parameter_dict['sampling_mode']  = parameter_dict.get('sampling_mode', 'simple')
    parameter_dict['fix_nonzero_prop'] = parameter_dict.get('fix_nonzero_prop', False)
    parameter_dict['nonzero_prop'] = parameter_dict.get('nonzero_prop', 0.9)
    if parameter_dict['next_cell_pred_type'] == 'identity':
      parameter_dict['next_weight'] = 0
    #mask_ratio = config.mask_ratio

    # n_input_bins = config.n_bins
    if parameter_dict['sampling_mode'] == 'hvg':
      parameter_dict['max_seq_len'] = parameter_dict['n_hvg'] + parameter_dict.get('non_hvg_size', 1000)+parameter_dict.get('append_cls', True)
    else:
      parameter_dict['max_seq_len'] = parameter_dict['n_hvg'] + parameter_dict.get('append_cls', True)

    use_wandb=True
    if use_wandb:
      run = wandb.init(
          config=parameter_dict,
          project=project_name,
          reinit=True,
          settings=wandb.Settings(start_method="fork"),
          #mode="online",
          mode=wandb_mode, #
      )
      config = wandb.config
    else:
      config = copy.deepcopy(parameter_dict)
      run=None
    print(config)

    set_seed(config.seed)


    if config.ADV and config.dab_weight >0:
        raise ValueError("ADV and DAB cannot be both True.")
    #DAB_separate_optim = True if config.dab_weight >0 else False

    return (config, run)

