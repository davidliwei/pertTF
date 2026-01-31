import os
import json
import torch
from typing import Optional, Any, Union, Dict
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download

# Ensure these are importable
from .pertTF import PerturbationTFModel
from ..utils.custom_tokenizer import SimpleVocab 
from .train_function import wrapper_train, eval_testdata

def legacy_vocab_loading(vocab_path):
    if vocab_path:
        import sys
        import types
        from pertTF.perttf.utils.custom_tokenizer import SimpleVocab  # Import your ACTUAL class

        # Define the legacy path that the file is looking for
        # (Based on your error: "No module named perttf")
        legacy_root = "perttf"
        legacy_full = "perttf.utils.custom_tokenizer"

        # 1. Create fake modules
        # We create the root 'perttf'
        fake_perttf = types.ModuleType(legacy_root)
        # We create 'perttf.utils'
        fake_utils = types.ModuleType(f"{legacy_root}.utils")
        # We create 'perttf.utils.custom_tokenizer'
        fake_tokenizer_mod = types.ModuleType(legacy_full)

        # 2. Link them together (so perttf.utils works)
        fake_perttf.utils = fake_utils
        fake_utils.custom_tokenizer = fake_tokenizer_mod

        # 3. PLANT YOUR CLASS inside the fake module
        # When pickle asks for 'SimpleVocab' from this module, it gets your class
        fake_tokenizer_mod.SimpleVocab = SimpleVocab

        # 4. Inject into sys.modules
        # This makes them "visible" to the import system
        sys.modules[legacy_root] = fake_perttf
        sys.modules[f"{legacy_root}.utils"] = fake_utils
        sys.modules[legacy_full] = fake_tokenizer_mod

        try:
            # 5. Load the file
            vocab_obj = torch.load(vocab_path, weights_only=False)
        except Exception as e:
            print(f"Error forcing vocab load: {e}")
        finally:
            # 6. Cleanup (Optional but recommended)
            # Remove the fake modules so they don't confuse the rest of your app
            if legacy_root in sys.modules: del sys.modules[legacy_root]
            if f"{legacy_root}.utils" in sys.modules: del sys.modules[f"{legacy_root}.utils"]
            if legacy_full in sys.modules: del sys.modules[legacy_full]
    return vocab_obj


class HFPerturbationTFModel(PerturbationTFModel, PyTorchModelHubMixin):
    def __init__(
        self,
        n_pert: int = 1,
        nlayers_pert: int = 4,
        n_ps: int = 1,
        ntoken: int = None,
        d_model: int = 32,
        nhead: int = 4,
        d_hid: int = None,
        nlayers: int = 2,
        nlayers_cls: int = 3,
        n_cls: int = 1,
        vocab: Any = None,
        dropout: float = 0.5,
        pad_token: str = "<pad>",
        pad_value: int = -2,
        do_mvc: bool = False,
        do_dab: bool = False,
        use_batch_labels: bool = False,
        num_batch_labels: Optional[int] = None,
        domain_spec_batchnorm: Union[bool, str] = False,
        input_emb_style: str = "continuous",
        n_bins: Optional[int] = 51,
        cell_emb_style: str = "cls",
        mvc_decoder_style: str = "inner product",
        ecs_threshold: float = 0.7,
        explicit_zero_prob: bool = False,
        use_fast_transformer: bool = False,
        fast_transformer_backend: str = "flash",
        pre_norm: bool = False,
        pred_lochness_next: bool = False,
        ps_decoder2_nlayer: int = 3,
        pert_pad_id: Optional[int] = None,
        pert_dim: Optional[int] = None,
        **kwargs
    ):
        # 1. Handle Training Config & Extras
        self.training_config = {}
        
        # Merge dedicated training_config if present
        if "training_config" in kwargs:
             self.training_config.update(kwargs.pop("training_config"))

        # Capture simple types from kwargs into training_config
        # We collect the keys to remove them later if we want a clean config
        training_keys = []
        for k, v in kwargs.items():
            if isinstance(v, (int, float, str, bool, type(None))):
                self.training_config[k] = v
                training_keys.append(k)

        # 2. Extract Specific Running Params
        if 'cell_type_to_index' in kwargs:
            self.cell_type_to_index = kwargs.pop('cell_type_to_index')
            n_cls = len(self.cell_type_to_index)
        
        if 'genotype_to_index' in kwargs:
            self.genotype_to_index = kwargs.pop('genotype_to_index')
            n_pert = len(self.genotype_to_index)
            
        if 'ps_names' in kwargs:
            self.ps_names = kwargs.pop('ps_names')
            n_ps = len(self.ps_names)

                # fix up some old configurations and param names
        if kwargs.get('layer_size', False):
            d_model = kwargs.pop('layer_size')

        if d_hid is None:
            d_hid = d_model

        if kwargs.get('GEPC', False):
            do_mvc = True
            
        #if config.get('embsize', False):
         #   config['d_model'] = config['layer_size']

        if kwargs.get('nheads', False):
            nhead = kwargs.pop('nheads')

        if kwargs.get('fast_transformer', False):
            use_fast_transformer = kwargs.pop('fast_transformer')

        ntoken = len(vocab) if vocab is not None else None
        # 3. Initialize Parent (Without **kwargs, as you requested)
        super().__init__(
            n_pert=n_pert,
            nlayers_pert=nlayers_pert,
            n_ps=n_ps,
            ntoken=ntoken,
            d_model=d_model,
            nhead=nhead,
            d_hid=d_hid,
            nlayers=nlayers,
            nlayers_cls=nlayers_cls,
            n_cls=n_cls,
            vocab=vocab,
            dropout=dropout,
            pad_token=pad_token,
            pad_value=pad_value,
            do_mvc=do_mvc,
            do_dab=do_dab,
            use_batch_labels=use_batch_labels,
            num_batch_labels=num_batch_labels,
            domain_spec_batchnorm=domain_spec_batchnorm,
            input_emb_style=input_emb_style,
            n_input_bins= n_bins,
            cell_emb_style=cell_emb_style,
            mvc_decoder_style=mvc_decoder_style,
            ecs_threshold=ecs_threshold,
            explicit_zero_prob=explicit_zero_prob,
            use_fast_transformer=use_fast_transformer,
            fast_transformer_backend=fast_transformer_backend,
            pre_norm=pre_norm,
            pred_lochness_next=pred_lochness_next,
            ps_decoder2_nlayer=ps_decoder2_nlayer,
            pert_pad_id=pert_pad_id,
            pert_dim=pert_dim,
            # Note: We do NOT pass **kwargs here, so parent doesn't see extra params.
        )

        # 4. SANITIZE HF CONFIG
        # The Mixin automatically captured EVERYTHING in __init__ into self.config.
        # If you want config.json to NOT contain training params, you must remove them here.
        if hasattr(self, "_hub_mixin_config"):
            # Remove vocab to prevent crash
            if "vocab" in self._hub_mixin_config:
                self._hub_mixin_config['ntoken'] = len(vocab)
                self._hub_mixin_config["vocab"] = None
            
            for n in list(self._hub_mixin_config.keys()):
                if type(self._hub_mixin_config[n]) in [dict, list]:
                    del self._hub_mixin_config[n]

            # Remove training params from the model config (Cleaner config.json)
            for k in training_keys:
                if k in self._hub_mixin_config:
                    del self._hub_mixin_config[k]
            
            # Remove the explicit 'training_config' dict if it was passed
            if "training_config" in self._hub_mixin_config:
                del self._hub_mixin_config["training_config"]
                
        self.training_config['pad_value'] = self._hub_mixin_config['pad_value']
        self.vocab = vocab

    def save_pretrained(self, save_directory: str, training_config: Optional[Dict] = None, **kwargs):
        super().save_pretrained(save_directory, **kwargs)

        # Save Vocab
        vocab_to_save = getattr(self, 'vocab', None)

        if vocab_to_save is not None:
            with open(os.path.join(save_directory, "vocab.json"), 'w') as json_file:
                json.dump(vocab_to_save.to_dict(), json_file) 
        

        # Save Running Params
        running_params_to_save = {
            'cell_type_to_index': getattr(self, 'cell_type_to_index', None),
            'genotype_to_index': getattr(self, 'genotype_to_index', None),
            'ps_names': getattr(self, 'ps_names', None),
            'num_batch_labels': getattr(self, 'num_batch_labels', 1)
        }
        # Filter None values
        running_params_to_save = {k: v for k, v in running_params_to_save.items() if v is not None}
        
        if running_params_to_save:
            torch.save(running_params_to_save, os.path.join(save_directory, "running_parameters.pt"))

        # Save Training Config
        final_train_config = self.training_config.copy() if self.training_config else {}
        if training_config:
            final_train_config.update(training_config)
            
        if final_train_config:
            with open(os.path.join(save_directory, "training_config.json"), "w") as f:
                json.dump(final_train_config, f, indent=2)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        def fetch_file(filename):
            if os.path.isdir(pretrained_model_name_or_path):
                file_path = os.path.join(pretrained_model_name_or_path, filename)
                return file_path if os.path.isfile(file_path) else None
            else:
                try:
                    return hf_hub_download(
                        repo_id=pretrained_model_name_or_path, 
                        filename=filename,
                        token=kwargs.get("token"), 
                        revision=kwargs.get("revision")
                    )
                except Exception:
                    return None

        # 1. LOAD CONFIG FIRST (Moved UP)
        # We must load this before we can assign anything to 'config'
        config_path = fetch_file("config.json")
        if not config_path:
            raise EnvironmentError(f"config.json not found in {pretrained_model_name_or_path}")

        with open(config_path, "r") as f:
            config = json.load(f)

        # 2. Load Training Config (And inject into config dict)
        train_cfg_path = fetch_file("training_config.json")
        if train_cfg_path:
            with open(train_cfg_path, "r") as f:
                # We pass this as a special key so __init__ can extract it
                config["training_config"] = json.load(f)

        # 3. Load Vocab
        old_vocab_obj = None
        vocab_path = fetch_file("vocab.pt")
        if vocab_path:
            old_vocab_obj = legacy_vocab_loading(vocab_path)
        else:
            vocab_path = fetch_file("vocab.json")
            if vocab_path:
                old_vocab_obj = SimpleVocab.from_json(vocab_path)
                       
        user_vocab = kwargs.get('vocab', None)
        if user_vocab is not None:
            vocab_merge = kwargs.pop('vocab_merge', 'custom')
            print(f'WARNING: user provide custom vocab, this is okay for finetuning, take the {vocab_merge} vocab')
            if vocab_merge == 'custom' or old_vocab_obj is None:
                active_vocab = user_vocab
            elif vocab_merge == 'union':
                active_vocab = user_vocab.stoi
                for k in old_vocab_obj.stoi:
                    if k not in active_vocab:
                        active_vocab[k] = len(active_vocab)
                active_vocab = SimpleVocab.from_dict(active_vocab)
            else:
                raise ValueError(f"vocab_merge is not one of custom or union")
        else:
            active_vocab = old_vocab_obj

        if active_vocab is None:
            raise EnvironmentError(f"vocab.pt or vocab.json not found in {pretrained_model_name_or_path}, not vocab provided by user")
        
        config["vocab"] = active_vocab
        if active_vocab:
            config["ntoken"] = len(active_vocab)

        # 4. Load Running Params
        running_params = {}
        running_param_path = fetch_file("running_parameters.pt")
        if running_param_path:
            running_params = torch.load(running_param_path, weights_only=False)
        
        # 5. Merge Parameters (Kwargs > RunningParams > Defaults)
        # Note: Fixed the 'kwargs(p_name)' syntax error here
        for p_name in ['genotype_to_index', 'cell_type_to_index']:
            if kwargs.get(p_name, False):
                print(f'WARNING: {p_name} provided by user, {p_name} related layers may be different from pretrained model, this is okay for finetuning')
                config[p_name] = kwargs[p_name]
            elif p_name in running_params:
                config[p_name] = running_params[p_name]
            # else: defaults handled by __init__ or logic below

        if kwargs.get('num_batch_labels', False) and type(kwargs.get('num_batch_labels', False)) == int:
            config['num_batch_labels'] = kwargs['num_batch_labels']
            print(f'WARNING: num_batch_labels provided by user, batch removal head may be different from pretrained model, this is okay for finetuning')
        elif 'num_batch_labels' in running_params:
            config['num_batch_labels'] = running_params['num_batch_labels']
            
        if kwargs.get('ps_names', False):
            config['ps_names'] = kwargs['ps_names']
            print(f'WARNING: ps column names provided by user, ps score prediction head may be different from pretrained model, this is okay for finetuning')
        elif 'ps_names' in running_params:
            config['ps_names'] = running_params['ps_names']


        # 6. Instantiate Model
        # If loading an OLD model, 'config' might contain training params (e.g., 'lr').
        # These will be passed to __init__, captured in **kwargs, and moved to self.training_config.
        model = cls(**config)

        # 7. Load Weights
        model_path = fetch_file("model.safetensors")
        if model_path:
            from safetensors.torch import load_file
            state_dict = load_file(model_path)
        else:
            bin_path = fetch_file("best_model.pt") 
            if bin_path:
                state_dict = torch.load(bin_path, weights_only=True)
                
        if state_dict is not None:
            # CALL THE REFACTORED WORKER FUNCTION
            loaded_layers = cls._smart_load_weights(
                model=model,
                state_dict=state_dict,
                old_vocab=old_vocab_obj,
                new_vocab=active_vocab
            )
            
            # Store the list of loaded layers in the model for freezing later
            model._loaded_layer_names = loaded_layers
            
            print(f"Model loaded. {len(loaded_layers)} layers transferred.")

        return model
    
    @staticmethod
    def _smart_load_weights(model, state_dict, old_vocab, new_vocab):
        """
        Loads state_dict into model, handling mismatches and performing 
        vocabulary embedding transfer if needed.
        """
        model_state_dict = model.state_dict()
        keys_to_drop = []
        loaded_keys = []

        # Check if we can perform embedding transfer
        # (Requires both vocabs and they must be different objects)
        do_vocab_transfer = (old_vocab is not None and new_vocab is not None and old_vocab is not new_vocab)
        
        for key in list(state_dict.keys()):
            if key not in model_state_dict:
                continue # Skip unknown keys

            param_new = model_state_dict[key]
            param_old = state_dict[key]

            # CASE 1: Exact Match
            if param_old.shape == param_new.shape:
                loaded_keys.append(key)
                continue

            # CASE 2: Shape Mismatch
            # Check if this is an embedding layer we can fix
            # usually named 'encoder.embedding.weight' or similar
            is_embedding = "embedding.weight" in key and param_new.dim() == 2
            
            if is_embedding and do_vocab_transfer:
                print(f"Attempting vocabulary transfer for layer: {key}")
                try:
                    # Create a new tensor with the NEW shape
                    new_weight = param_new.clone().detach() # Start with random init of current model
                    
                    # Calculate intersection of tokens
                    # Assuming vocabs have .stoi (string to index)
                    common_tokens = set(old_vocab.stoi.keys()) & set(new_vocab.stoi.keys())
                    
                    transferred_count = 0
                    for token in common_tokens:
                        old_idx = old_vocab.stoi[token]
                        new_idx = new_vocab.stoi[token]
                        
                        # Copy the vector
                        new_weight[new_idx] = param_old[old_idx]
                        transferred_count += 1
                    
                    # Update state_dict with the grafted weight
                    state_dict[key] = new_weight
                    loaded_keys.append(key) # We count this as "loaded" since we transferred info
                    
                    print(f" - Transferred {transferred_count}/{len(new_vocab)} tokens.")
                    continue 

                except Exception as e:
                    print(f" - Vocab transfer failed for {key}: {e}")
                    # Fall through to drop

            # CASE 3: Unresolvable Mismatch -> Drop
            print(f"Dropping layer {key} due to shape mismatch: {param_old.shape} vs {param_new.shape}")
            keys_to_drop.append(key)

        # Cleanup state_dict
        for key in keys_to_drop:
            del state_dict[key]

        # Load
        model.load_state_dict(state_dict, strict=False)
        
        return loaded_keys

    # ----------------------------------------------------------------------
    # UTILITY: Freeze Loaded Layers
    # ----------------------------------------------------------------------
    def freeze_pretrained_layers(self):
        """
        Freezes all parameters that were successfully loaded from the checkpoint.
        New heads or mismatched layers remain trainable.
        """
        if not hasattr(self, '_loaded_layer_names'):
            print("Warning: No loaded layer record found. Cannot freeze specific layers.")
            return

        frozen_count = 0
        for name, param in self.named_parameters():
            if name in self._loaded_layer_names:
                param.requires_grad = False
                frozen_count += 1
        
        print(f"Froze {frozen_count} pretrained parameters.")

    # ----------------------------------------------------------------------
    # UTILITY: Enforce a anndata object to be compatible with the model
    # ----------------------------------------------------------------------
    # TODO: Finish these functions for user demo usage
    def comply_anndata(self, anndata, celltype_col = 'celltype', genotype_col ='genotype'):
        print('Force Complying anndata object with model, use this only for inference on test data, it WILL alter the anndata object')
        pass

    def _init_default_train_config_(self):
        # initiate default training configuration
        pass

    def run_train(self, anndata):
        pass

    def eval_identity(self, anndata):
        pass  

    def eval_perturb(self, anndata):
        pass

    def eval_lochness(self, anndata):
        pass

    def run_test(self, anndata):
        pass

