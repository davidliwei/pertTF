import os
import json
import torch
from pathlib import Path
from typing import Optional, Any, Union, Dict
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download
from omegaconf import OmegaConf

# Ensure these are importable
from .pertTF import PerturbationTFModel
from ..utils.custom_tokenizer import SimpleVocab 


def legacy_vocab_loading(vocab_path):
    if not vocab_path:
        return None
    import pickle
    import types

    class VocabUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            # Resolve vocabularies saved through either supported import namespace.
            if name == "SimpleVocab" and module in {
                "perttf.utils.custom_tokenizer",
                "pertTF.perttf.utils.custom_tokenizer",
            }:
                return SimpleVocab
            return super().find_class(module, name)

    vocab_pickle = types.ModuleType("perttf_vocab_pickle")
    vocab_pickle.__dict__.update(vars(pickle))
    vocab_pickle.Unpickler = VocabUnpickler
    return torch.load(vocab_path, weights_only=False, pickle_module=vocab_pickle)


class HFPerturbationTFModel(PerturbationTFModel, PyTorchModelHubMixin):
    def __init__(
        self,
        n_pert: int = 1,
        nlayers_pert: int = 4,
        n_ps: Optional[int] = None,
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
        distribution: Optional[str] = None,
        pert_sources: Optional[Dict] = None,
        control_label: str = "WT",
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
        
        # 2. Extract Specific Running Params to accomodate legacy saved models and configs
        if 'cell_type_to_index' in kwargs:
            self.cell_type_to_index = kwargs.pop('cell_type_to_index')
            self._hub_mixin_config['n_cls'] = len(self.cell_type_to_index)
        
        if 'genotype_to_index' in kwargs:
            self.genotype_to_index = kwargs.pop('genotype_to_index')
            self._hub_mixin_config['n_pert'] = len(self.genotype_to_index)
            
        if 'ps_names' in kwargs:
            self.ps_names = kwargs.pop('ps_names')

        if n_ps is None:
            n_ps = len(self.ps_names) if hasattr(self, 'ps_names') else 1
        self._hub_mixin_config['n_ps'] = n_ps

        # Optional perturbation FEATURE matrix (n_pert, feat_dim). Like the other
        # running params it is a tensor/array (not JSON-serializable), so it is kept
        # out of the HF config and forwarded directly to the parent, which builds a
        # FeaturePertEncoder when it is not None (enables zero-shot prediction).
        self.pert_features = kwargs.pop('pert_features', None)
        if pert_sources is not None and self.pert_features is not None:
            raise ValueError("Choose pert_sources or legacy pert_features, not both")

                # fix up some old configurations and param names
        if kwargs.get('layer_size', False):
            self._hub_mixin_config['d_model'] = kwargs.pop('layer_size')

        if d_hid is None:
            self._hub_mixin_config['d_hid'] = self._hub_mixin_config['d_model']

        if kwargs.get('GEPC', False):
            self._hub_mixin_config['do_mvc'] = True
            
        #if config.get('embsize', False):
         #   config['d_model'] = config['layer_size']

        if kwargs.get('nheads', False):
            self._hub_mixin_config['nhead'] = kwargs.pop('nheads')

        if kwargs.get('fast_transformer', False):
            self._hub_mixin_config['use_fast_transformer'] = kwargs.pop('fast_transformer')

        if kwargs.get('dab_weight', 0.0) > 0:
            self._hub_mixin_config['do_dab'] = True

        if kwargs.get('pred_lochness_next', 0) > 0:
            self._hub_mixin_config['pred_lochness_next'] = kwargs['pred_lochness_next']

        ntoken = len(vocab) if vocab is not None else None
        # 3. Initialize Parent (Without **kwargs, as you requested)
        super().__init__(
            n_pert=self._hub_mixin_config['n_pert'],
            nlayers_pert=nlayers_pert,
            n_ps=self._hub_mixin_config['n_ps'],
            ntoken=ntoken,
            d_model=self._hub_mixin_config['d_model'],
            nhead=self._hub_mixin_config['nhead'],
            d_hid=self._hub_mixin_config['d_hid'],
            nlayers=nlayers,
            nlayers_cls=nlayers_cls,
            n_cls=self._hub_mixin_config['n_cls'],
            vocab=vocab,
            dropout=dropout,
            pad_token=pad_token,
            pad_value=pad_value,
            do_mvc=self._hub_mixin_config['do_mvc'],
            do_dab=self._hub_mixin_config['do_dab'],
            use_batch_labels=use_batch_labels,
            num_batch_labels=num_batch_labels,
            domain_spec_batchnorm=domain_spec_batchnorm,
            input_emb_style=input_emb_style,
            n_input_bins= n_bins,
            cell_emb_style=cell_emb_style,
            mvc_decoder_style=mvc_decoder_style,
            ecs_threshold=ecs_threshold,
            explicit_zero_prob=explicit_zero_prob,
            distribution=distribution,
            use_fast_transformer=self._hub_mixin_config['use_fast_transformer'],
            fast_transformer_backend=fast_transformer_backend,
            pre_norm=pre_norm,
            pred_lochness_next=self._hub_mixin_config['pred_lochness_next'],
            ps_decoder2_nlayer=ps_decoder2_nlayer,
            pert_pad_id=pert_pad_id,
            pert_dim=pert_dim,
            pert_features=self.pert_features,
            # Note: We do NOT pass **kwargs here, so parent doesn't see extra params.
        )

        if pert_sources is not None:
            from .pert_encoder import UnifiedPertEncoder
            if not hasattr(self, 'genotype_to_index'):
                raise ValueError("Unified encoder requires the expanded genotype_to_index")
            self.pert_encoder = UnifiedPertEncoder(
                pert_sources, self.genotype_to_index,
                self.d_model if pert_dim is None else pert_dim,
                padding_idx=pert_pad_id, control_label=control_label,
            )

        # 4. SANITIZE HF CONFIG
        # The Mixin automatically captured EVERYTHING in __init__ into self.config.
        # If you want config.json to NOT contain training params, you must remove them here.
        # Remove vocab from the hub config: it is not JSON-serializable, and
        # leaving the key (even as None) leaks into self.training_config, which
        # is later unpacked into PertBatchCollator(vocab, ..., **config).
        if "vocab" in self._hub_mixin_config:
            self._hub_mixin_config['ntoken'] = len(vocab)
            del self._hub_mixin_config["vocab"]
        
        for n in list(self._hub_mixin_config.keys()):
            if type(self._hub_mixin_config[n]) in [dict, list]:
                del self._hub_mixin_config[n]
        self._hub_mixin_config.pop("pert_sources", None)

        # Remove training params from the model config (Cleaner config.json)
        for k in training_keys:
            if k in self._hub_mixin_config:
                del self._hub_mixin_config[k]
        
        # Remove the explicit 'training_config' dict if it was passed
        if "training_config" in self._hub_mixin_config:
            del self._hub_mixin_config["training_config"]
                
        #self.training_config['pad_value'] = self._hub_mixin_config['pad_value']
        self.vocab = vocab
        self.training_config.update(self._hub_mixin_config)
        self.training_config = OmegaConf.create(self.training_config)

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
            'num_batch_labels': getattr(self, 'num_batch_labels', 1),
            # Needed at from_pretrained time to rebuild FeaturePertEncoder BEFORE
            # weights load, otherwise pert_encoder.proj/features keys are dropped.
            'pert_features': getattr(self, 'pert_features', None),
        }
        # Filter None values
        running_params_to_save = {k: v for k, v in running_params_to_save.items() if v is not None}
        from .pert_encoder import UnifiedPertEncoder
        if isinstance(self.pert_encoder, UnifiedPertEncoder):
            running_params_to_save['pert_source_config'] = self.pert_encoder.source_config()
        
        if running_params_to_save:
            torch.save(running_params_to_save, os.path.join(save_directory, "running_parameters.pt"))

        # Save Training Config
        final_train_config = dict(self.training_config) if self.training_config else {}
        if training_config:
            final_train_config.update(training_config)
            
        if final_train_config:
            with open(os.path.join(save_directory, "training_config.json"), "w") as f:
                json.dump(final_train_config, f, indent=2)

    @property
    def prediction_scale(self) -> str:
        """Native scale returned by the expression distribution generator."""
        distribution = getattr(self, "distribution", None)
        return "log1p" if distribution in {None, "zig", "gaussian"} else "counts"

    def predict_perturbations(
        self,
        adata,
        *,
        input_layer: str = "X_binned",
        perturbation_col: str = "genotype_next",
        prediction_mode: str = "sample",
        prediction_seed: Optional[int] = 0,
        gene_sampling_mode: Optional[str] = None,
        max_seq_len: Optional[int] = None,
        use_size_factor: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        query_sources: Optional[Dict] = None,
    ):
        """Predict row-level perturbations already assigned in an AnnData object.

        Simple sampling uses max_seq_len as a token limit. Expressed sampling
        allows at least 10,000 genes, bounded by the available genes. HVG
        sampling retains all annotated HVGs; an explicit max_seq_len above the
        HVG count sets the total gene budget, otherwise non_hvg_size is retained.
        Expressed and HVG budgets reserve an additional slot for the CLS token.
        Prediction metadata records the resolved token limit.
        With a unified encoder, query_sources supplies temporary (genes, matrix)
        tuples for new target IDs through existing source projections. Registered
        IDs cannot be overridden, and no model vocabulary or weights are changed.
        """
        from .train_function import eval_testdata

        if prediction_mode not in {"mean", "sample"}:
            raise ValueError("prediction_mode must be 'mean' or 'sample'")
        if perturbation_col != "genotype_next":
            raise ValueError("perturbation_col must be 'genotype_next'")
        if input_layer not in adata.layers:
            raise ValueError(f"AnnData is missing expression layer {input_layer!r}")
        required_obs = {"celltype", "genotype", perturbation_col}
        missing_obs = required_obs.difference(adata.obs.columns)
        if missing_obs:
            raise ValueError(f"AnnData is missing required obs columns: {sorted(missing_obs)}")
        if not adata.var_names.is_unique:
            raise ValueError("AnnData var_names must be unique")
        if getattr(self, "vocab", None) is None:
            raise RuntimeError("Loaded model does not contain a gene vocabulary")
        if not hasattr(self, "genotype_to_index"):
            raise RuntimeError("Loaded model does not contain genotype_to_index")
        if not hasattr(self, "cell_type_to_index"):
            raise RuntimeError("Loaded model does not contain cell_type_to_index")

        for column, mapping in (
            ("celltype", self.cell_type_to_index),
            ("genotype", self.genotype_to_index),
        ):
            values = adata.obs[column]
            if values.isna().any():
                raise ValueError(f"AnnData obs[{column!r}] contains missing values")
            missing_labels = sorted(set(values).difference(mapping))
            if missing_labels:
                raise ValueError(
                    f"AnnData obs[{column!r}] contains labels absent from the model mapping: "
                    f"{missing_labels[:10]}"
                )
        missing_genes = [gene for gene in adata.var_names if gene not in self.vocab.stoi]
        if missing_genes:
            preview = ", ".join(map(str, missing_genes[:10]))
            raise ValueError(
                f"{len(missing_genes)} inference genes are absent from the model vocabulary: {preview}"
            )
        perturbations = adata.obs[perturbation_col]
        if perturbations.isna().any():
            raise ValueError(f"AnnData obs[{perturbation_col!r}] contains missing values")
        from .pert_encoder import UnifiedPertEncoder
        unified = isinstance(self.pert_encoder, UnifiedPertEncoder)
        registered = self.genotype_to_index
        query_ids = set()
        if query_sources is not None:
            if not unified:
                raise ValueError("query_sources requires the unified perturbation encoder")
            # Detailed table validation and collision checks run in encode_queries.
            query_ids = {gene for genes, _ in query_sources.values() for gene in genes}
        missing_perturbations = sorted(set(perturbations).difference(registered).difference(query_ids))
        if missing_perturbations:
            raise ValueError(
                "Inference perturbations are absent from genotype_to_index: "
                f"{missing_perturbations[:10]}"
            )

        config = self._init_default_train_config_()
        if "vocab" in config:
            del config["vocab"]
        config.next_cell_pred_type = "pert"
        config.use_batch_label = bool(self.use_batch_labels or self.domain_spec_batchnorm)
        if gene_sampling_mode is not None:
            if gene_sampling_mode not in {"simple", "expressed", "hvg"}:
                raise ValueError("gene_sampling_mode must be 'simple', 'expressed', or 'hvg'")
            config.sampling_mode = gene_sampling_mode
        if max_seq_len is not None:
            if int(max_seq_len) <= 0:
                raise ValueError("max_seq_len must be positive")
            config.max_seq_len = int(max_seq_len)

        append_cls = int(config.get("append_cls", True))
        if config.sampling_mode == "expressed":
            n_genes = min(adata.n_vars, max(10000, int(config.max_seq_len)))
            config.max_seq_len = n_genes + append_cls
        elif config.sampling_mode == "hvg":
            hvg_col = config.get("hvg_col", "highly_variable")
            if hvg_col not in adata.var:
                raise ValueError(f"AnnData var is missing HVG column {hvg_col!r}")
            n_hvg = int(adata.var[hvg_col].sum())
            non_hvg_size = int(config.get("non_hvg_size", 1000))
            if max_seq_len is not None and int(max_seq_len) > n_hvg:
                non_hvg_size = int(max_seq_len) - n_hvg
            config.non_hvg_size = min(non_hvg_size, adata.n_vars - n_hvg)
            config.max_seq_len = n_hvg + config.non_hvg_size + append_cls
        else:
            config.max_seq_len = min(int(config.max_seq_len), adata.n_vars + append_cls)
        # The inference sampling policy takes precedence over training-time full tokenization.
        config.full_tokenize = False

        if device is None:
            device = next(self.parameters()).device
        device = torch.device(device)
        self.to(device)
        self.eval()
        inference_mapping = self.genotype_to_index
        perturbation_embeddings = None
        if query_sources is not None:
            # Registered targets use the dataset/model mapping directly. Only
            # temporary query IDs need an evaluation-local extension and vectors.
            inference_mapping = dict(self.genotype_to_index)
            for gene in perturbations:
                if gene not in inference_mapping:
                    inference_mapping[gene] = len(inference_mapping)
            with torch.no_grad():
                names, vectors = self.pert_encoder.encode_queries(query_sources)
                query_vectors = dict(zip(names, vectors))
                perturbation_embeddings = next(self.parameters()).new_zeros(
                    len(inference_mapping), self.pert_encoder.embedding_dim)
                requested = list(dict.fromkeys(perturbations))
                known = [gene for gene in requested if gene in registered]
                if known:
                    indices = torch.tensor([registered[g] for g in known], device=device)
                    rows = torch.tensor([inference_mapping[g] for g in known], device=device)
                    perturbation_embeddings[rows] = self.pert_encoder(indices)
                for gene in requested:
                    if gene in query_vectors:
                        perturbation_embeddings[inference_mapping[gene]] = query_vectors[gene]
        result = eval_testdata(
            self,
            adata,
            list(adata.var_names),
            {
                "cell_type_to_index": self.cell_type_to_index,
                "genotype_to_index": inference_mapping,
                "vocab": self.vocab,
            },
            config,
            input_layer_key=input_layer,
            make_plots=False,
            predict_expr=True,
            mvc_full_expr=True,
            sizefactor=use_size_factor,
            sample=prediction_mode == "sample",
            sample_seed=prediction_seed,
            device=device,
            max_seq_len=int(config.max_seq_len),
            perturbation_embeddings=perturbation_embeddings,
        )
        if not result.obs_names.equals(adata.obs_names):
            raise RuntimeError("pertTF inference did not preserve the requested rows and their order")
        native = result.obsm.get("mvc_next_expr")
        if native is None or native.shape != result.shape:
            raise RuntimeError(
                "pertTF inference did not return mvc_next_expr aligned to the requested genes"
            )
        result.uns["perttf_prediction"] = {
            "distribution": getattr(self, "distribution", None),
            "native_scale": self.prediction_scale,
            "input_layer": input_layer,
            "perturbation_col": perturbation_col,
            "prediction_mode": prediction_mode,
            "prediction_seed": prediction_seed,
            "gene_sampling_mode": str(config.sampling_mode),
            "max_seq_len": int(config.max_seq_len),
        }
        if unified:
            result.uns["perttf_prediction"]["perturbation_sources"] = list(self.pert_encoder.sources)
            result.uns["perttf_prediction"]["query_perturbations"] = sorted(query_ids)
        if config.sampling_mode == "hvg":
            result.uns["perttf_prediction"]["non_hvg_size"] = int(config.non_hvg_size)
        return result

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        """Restore HF-saved inference models with strict weight loading by default.

        Legacy checkpoints use constructor defaults for missing architecture
        settings and fail if their weights do not match. Use strict=False only
        for intentional partial weight transfer, such as fine-tuning.
        """
        strict = kwargs.pop("strict", True)

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
            if strict and old_vocab_obj is not None and user_vocab.to_dict() != old_vocab_obj.to_dict():
                raise ValueError("strict loading does not allow replacing the checkpoint vocabulary")
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
            if kwargs.get(p_name, None) is not None:
                if strict and p_name in running_params and kwargs[p_name] != running_params[p_name]:
                    raise ValueError(f"strict loading does not allow replacing {p_name}")
                print(f'WARNING: {p_name} provided by user, {p_name} related layers may be different from pretrained model, this is okay for finetuning')
                config[p_name] = kwargs[p_name]
            elif p_name in running_params:
                config[p_name] = running_params[p_name]
            # else: defaults handled by __init__ or logic below

        if kwargs.get('num_batch_labels', None) is not None and type(kwargs['num_batch_labels']) == int:
            if strict and 'num_batch_labels' in running_params and kwargs['num_batch_labels'] != running_params['num_batch_labels']:
                raise ValueError("strict loading does not allow replacing num_batch_labels")
            config['num_batch_labels'] = kwargs['num_batch_labels']
            print(f'WARNING: num_batch_labels provided by user, batch removal head may be different from pretrained model, this is okay for finetuning')
        elif 'num_batch_labels' in running_params:
            config['num_batch_labels'] = running_params['num_batch_labels']
            
        if kwargs.get('ps_names', None) is not None:
            if strict and 'ps_names' in running_params and kwargs['ps_names'] != running_params['ps_names']:
                raise ValueError("strict loading does not allow replacing ps_names")
            config['ps_names'] = kwargs['ps_names']
            if not strict:
                config['n_ps'] = len(kwargs['ps_names'])
            print(f'WARNING: ps column names provided by user, ps score prediction head may be different from pretrained model, this is okay for finetuning')
        elif 'ps_names' in running_params:
            config['ps_names'] = running_params['ps_names']

        # Legacy runs always recorded a placeholder PS name, including models
        # trained with ps_weight=0 and therefore no PS decoder parameters.
        if 'n_ps' not in config:
            ps_weight = config.get('ps_weight', config.get('training_config', {}).get('ps_weight', 0))
            config['n_ps'] = len(config.get('ps_names', [])) if ps_weight > 0 else 0

        # Restore the perturbation feature matrix so the parent builds a
        # FeaturePertEncoder before weights are loaded (zero-shot encoder).
        if kwargs.get('pert_features', None) is not None:
            if strict and running_params.get('pert_features', None) is not None:
                if not torch.equal(torch.as_tensor(kwargs['pert_features']), torch.as_tensor(running_params['pert_features'])):
                    raise ValueError("strict loading does not allow replacing pert_features")
            config['pert_features'] = kwargs['pert_features']
            print(f'WARNING: pert_features provided by user, perturbation encoder may be different from pretrained model, this is okay for finetuning')
        elif running_params.get('pert_features', None) is not None:
            config['pert_features'] = running_params['pert_features']

        # let user option to choose attention backend
        if 'use_fast_transformer' in kwargs:
            config['use_fast_transformer'] = kwargs['use_fast_transformer']
            config['fast_transformer'] = config['use_fast_transformer']

        if 'fast_transformer_backend' in kwargs:
            config['fast_transformer_backend'] = kwargs['fast_transformer_backend']
        # Load tensors before reconstruction so unified feature buffers need not
        # also be duplicated in running_parameters.pt or downloaded from HF.
        state_dict = None
        model_path = fetch_file("model.safetensors")
        if model_path:
            from safetensors.torch import load_file
            state_dict = load_file(model_path)
        else:
            bin_path = fetch_file("best_model.pt") 
            if bin_path:
                state_dict = torch.load(bin_path, weights_only=True, map_location=torch.device('cpu'))
                
        if state_dict is None:
            raise EnvironmentError(
                f"model.safetensors or best_model.pt not found in {pretrained_model_name_or_path}"
            )

        source_config = running_params.get('pert_source_config')
        if kwargs.get('pert_sources') is not None:
            raise ValueError("Changing perturbation sources during checkpoint loading is not supported")
        if source_config is not None:
            if not strict:
                raise ValueError("Unified encoder checkpoints currently require strict=True")
            config['control_label'] = source_config['control_label']
            config['pert_sources'] = {
                name: (table['genes'], state_dict[f'pert_encoder.sources.{name}.features'])
                for name, table in source_config['sources'].items()
            }
        model = cls(**config)

        if strict:
            loaded_layers = cls._strict_load_weights(model, state_dict)
        else:
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
    def _strict_load_weights(model, state_dict):
        """Load every checkpoint tensor after deterministic attention-key remapping."""
        model_keys = set(model.state_dict())
        remappings = (
            ("self_attn.in_proj_weight", "qkv_proj.weight"),
            ("self_attn.in_proj_bias", "qkv_proj.bias"),
            ("self_attn.out_proj.", "out_proj."),
        )
        remapped = {}
        for key, value in state_dict.items():
            target_key = key
            if target_key not in model_keys:
                for left, right in remappings:
                    candidate = None
                    if left in key:
                        candidate = key.replace(left, right)
                    elif right in key:
                        candidate = key.replace(right, left)
                    if candidate in model_keys:
                        target_key = candidate
                        break
            if target_key in remapped:
                raise RuntimeError(f"Multiple checkpoint tensors map to {target_key}")
            remapped[target_key] = value

        model.load_state_dict(remapped, strict=True)
        return list(remapped)
    
    @staticmethod
    def _smart_load_weights(model, state_dict, old_vocab, new_vocab):
        """
        Loads state_dict into model, handling mismatches, performing 
        vocabulary embedding transfer, and mapping between Vanilla/Flash layers.
        """
        model_state_dict = model.state_dict()
        keys_to_drop = []
        loaded_keys = []

        # Check if we can perform embedding transfer
        do_vocab_transfer = (old_vocab is not None and new_vocab is not None and old_vocab is not new_vocab)
        
        # Define bidirectional replacement rules for Vanilla <-> Flash mapping
        # Format: (Pattern A, Pattern B) - will try replacing A with B and B with A
        remappings = [
            ("self_attn.in_proj_weight", "qkv_proj.weight"),
            ("self_attn.in_proj_bias",   "qkv_proj.bias"),
            ("self_attn.out_proj.",       "out_proj.") 
        ]

        # Iterate over a copy of keys so we can modify state_dict if needed
        for key in list(state_dict.keys()):
            
            # --- STEP 0: Key Remapping (Vanilla <-> Flash) ---
            target_key = key
            
            # If the exact key isn't in the model, try to find a remapped equivalent
            if key not in model_state_dict:
                for pat_a, pat_b in remappings:
                    if pat_a in key:
                        potential_key = key.replace(pat_a, pat_b)
                        if potential_key in model_state_dict:
                            print(f"Remapping (Vanilla->Fast):{key} -> {potential_key}")
                            target_key = potential_key
                            break
                    elif pat_b in key:
                        potential_key = key.replace(pat_b, pat_a)
                        if potential_key in model_state_dict:
                            print(f"Remapping (Fast->Vanilla):{key} -> {potential_key}")
                            target_key = potential_key
                            break

            # If after remapping we still don't have a match, skip it
            if target_key not in model_state_dict:
                print(f"Skipping unknown key: {key}") 
                continue 

            param_old = state_dict[key]
            param_new = model_state_dict[target_key]

            # If we remapped, we must update the state_dict to use the NEW key
            # so load_state_dict(strict=False) picks it up later.
            if target_key != key:
                state_dict[target_key] = param_old
                keys_to_drop.append(key) # Mark old key for deletion

            # CASE 1: Exact Match
            if param_old.shape == param_new.shape:
                loaded_keys.append(target_key)
                continue

            # CASE 2: Shape Mismatch
            # Check if this is an embedding layer we can fix
            # usually named 'encoder.embedding.weight' or similar
            is_embedding = "embedding.weight" in target_key and param_new.dim() == 2
            
            if is_embedding and do_vocab_transfer:
                print(f"Attempting vocabulary transfer for layer: {target_key}")
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
                    state_dict[target_key] = new_weight
                    loaded_keys.append(target_key)
                    
                    print(f" - Transferred {transferred_count}/{len(new_vocab)} tokens.")
                    continue 

                except Exception as e:
                    print(f" - Vocab transfer failed for {target_key}: {e}")
            
            # --- STEP 3: Unresolvable Mismatch -> Drop ---
            print(f"Dropping layer {target_key} due to shape mismatch: {param_old.shape} vs {param_new.shape}")
            keys_to_drop.append(target_key)

        # Cleanup state_dict
        for key in keys_to_drop:
            if key in state_dict:
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
        fallback_defaults = {
            "seed": 42,
            "batch_size": 8,
            "special_tokens": ["<pad>", "<cls>", "<eos>", "<unk>"],
            "n_bins": getattr(self, "n_input_bins", 51) or 51,
            "n_hvg": 3000,
            "max_seq_len": 3000,
            "sampling_mode": "simple",
            "append_cls": True,
            "include_zero_gene": True,
            "mask_ratio": 0.15,
            "mask_value": -1,
            "pad_value": getattr(self, "pad_value", -2),
            "pad_token": "<pad>",
            "lr": 1e-3,
            "schedule_ratio": 0.99,
            "schedule_interval": 1,
            "lr_ADV": 1e-3,
            "amp": False,
            "amp_dtype": "fp16",
            "log_interval": 10,
            "layer_size": getattr(self, "d_model", 32),
            "do_train": True,
            "GEPC": False,
            "CCE": False,
            "ADV": False,
            "DSBN": False,
            "use_batch_label": False,
            "perturbation_input": False,
            "cell_type_classifier": True,
            "genotype_classifier": True,
            "cell_type_classifier_weight": 1.0,
            "perturbation_classifier_weight": 1.0,
            "this_weight": 1.0,
            "next_weight": 0.0,
            "next_cell_pred_type": "identity",
            "ecs_thres": 0.0,
            "ecs_weight": 1.0,
            "dab_weight": 0.0,
            "ps_weight": 0.0,
            "explicit_zero_prob": getattr(self, "explicit_zero_prob", False),
            "distribution": None,
            "use_ot": False,
            "dataset_name": "adata",
        }

        base_config = self.training_config if self.training_config is not None else OmegaConf.create({})
        merged = OmegaConf.merge(OmegaConf.create(fallback_defaults), base_config)
        return merged

    @staticmethod
    def build_lora_config(r=8, lora_alpha=32, lora_dropout=0.1, target_modules=None):
        try:
            from peft import LoraConfig
        except ImportError as e:
            raise ImportError(
                "The 'peft' package is required to use LoRA-related functionality. "
                "Install it with `pip install peft` and try again."
            ) from e

        if target_modules is None:
            target_modules = [
                "qkv_proj",
                "out_proj",
                "linear1",
                "linear2",
                "decoder.fc.0",
                "decoder.fc.2",
            ]
        return LoraConfig(
            inference_mode=False,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
        )
        
    def run_lora_train(
        self,
        adata,
        epochs: int = 1,
        batch_size: Optional[int] = None,
        lr: Optional[float] = None,
        train_val_split: float = 0.2,
        input_layer_key: str = "X_binned",
        next_layer_key: str = "X_binned_next",
        lora_config=None,
        device: Optional[torch.device] = None,
        save_dir: Optional[str] = None,
        amp: Optional[bool] = None,
        amp_dtype: Optional[str] = None,
        log_interval: Optional[int] = None,
        seed: Optional[int] = None,
        train_indices=None,
        valid_indices=None,
        stratify_by=None,
        split_check_columns=None,
    ):
        import random
        import numpy as np
        from peft import get_peft_model
        from . import train_function
        from .train_data_gen import produce_training_datasets
        from ..utils.set_optimizer import create_optimizer_dict

        if "genotype" not in adata.obs.columns and "condition" in adata.obs.columns:
            adata.obs["genotype"] = adata.obs["condition"].copy()
        if input_layer_key not in adata.layers:
            adata.layers[input_layer_key] = adata.X.copy()
        if next_layer_key not in adata.layers:
            adata.layers[next_layer_key] = adata.layers[input_layer_key].copy()

        config = self._init_default_train_config_()
        # Avoid duplicate kwarg collision in PertBatchCollator(vocab=..., **config).
        if "vocab" in config:
            del config["vocab"]
        # Keep training flags aligned with the loaded model, not stale checkpoint training_config.
        config.GEPC = hasattr(self, "mvc_decoder")
        config.explicit_zero_prob = bool(getattr(self, "explicit_zero_prob", False))
        config.distribution = getattr(self, "distribution", None)
        if batch_size is not None:
            config.batch_size = batch_size
        if lr is not None:
            config.lr = lr
        if amp is not None:
            config.amp = amp
        if amp_dtype is not None:
            config.amp_dtype = amp_dtype
        if log_interval is not None:
            config.log_interval = log_interval
        config.dataset_name = Path(getattr(adata, "filename", "") or "adata").stem

        # Reproducibility: seed all RNGs before data loading and training.
        effective_seed = seed if seed is not None else config.get("seed", 42)
        random.seed(effective_seed)
        np.random.seed(effective_seed)
        torch.manual_seed(effective_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(effective_seed)

        ps_columns = getattr(self, "ps_names", None)
        if ps_columns:
            valid_ps_columns = [col for col in ps_columns if col in adata.obs.columns]
            if len(valid_ps_columns) != len(ps_columns):
                print(
                    f"[run_lora_train] filtered ps columns to existing obs columns: "
                    f"{valid_ps_columns if valid_ps_columns else 'None'}"
                )
            ps_columns = valid_ps_columns if valid_ps_columns else None

        cell_type_to_index = getattr(self, "cell_type_to_index", None)
        if cell_type_to_index is not None and "celltype" in adata.obs.columns:
            missing_celltypes = [x for x in adata.obs["celltype"].unique() if x not in cell_type_to_index]
            if missing_celltypes:
                print(
                    "[run_lora_train] model cell_type_to_index missing labels from current adata; "
                    "rebuilding celltype mapping from adata."
                )
                cell_type_to_index = None

        genotype_to_index = getattr(self, "genotype_to_index", None)
        if genotype_to_index is not None and "genotype" in adata.obs.columns:
            missing_genotypes = [x for x in adata.obs["genotype"].unique() if x not in genotype_to_index]
            if missing_genotypes:
                print(
                    "[run_lora_train] model genotype_to_index missing labels from current adata; "
                    "rebuilding genotype mapping from adata."
                )
                genotype_to_index = None

        data_gen = produce_training_datasets(
            adata_input=adata,
            config=config,
            input_layer_key=input_layer_key,
            next_layer_key=next_layer_key,
            next_cell_pred=config.next_cell_pred_type,
            cell_type_to_index=cell_type_to_index,
            genotype_to_index=genotype_to_index,
            vocab=self.vocab,
            ps_columns=ps_columns,
            train_val_split=train_val_split,
            train_indices=train_indices,
            valid_indices=valid_indices,
            stratify_by=stratify_by,
            split_check_columns=split_check_columns,
        )

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.to(device)
        peft_model = get_peft_model(self, lora_config or self.build_lora_config()).to(device)

        # train_function logs through wandb by default; keep this entry-point side-effect free.
        if hasattr(train_function, "wandb") and hasattr(train_function.wandb, "log"):
            train_function.wandb.log = lambda *args, **kwargs: None

        optimizer_dict = create_optimizer_dict(peft_model, device, config, data_gen["num_batch_types"])

        best_val_loss = float("inf")
        best_epoch = 0
        best_state_dict = None

        for epoch in range(1, epochs + 1):
            train_function.train(
                model=peft_model,
                loader=data_gen["train_loader"],
                config=config,
                vocab=data_gen["vocab"],
                optim_dict=optimizer_dict,
                epoch=epoch,
                logger=None,
                device=device,
            )

            # Validation
            val_results = train_function.evaluate(
                model=peft_model,
                loader=data_gen["valid_loader"],
                config=config,
                vocab=data_gen["vocab"],
                epoch=epoch,
                device=device,
            )
            val_mse = val_results[0]
            print(f"Epoch {epoch}/{epochs} | val_mse: {val_mse:.4f}")

            if val_mse < best_val_loss:
                best_val_loss = val_mse
                best_epoch = epoch
                best_state_dict = {k: v.cpu().clone() for k, v in peft_model.state_dict().items()}

        # Restore best checkpoint
        if best_state_dict is not None:
            peft_model.load_state_dict(best_state_dict)
            peft_model.to(device)
            print(f"Restored best model from epoch {best_epoch} (val_mse={best_val_loss:.4f})")

        if save_dir:
            adapter_dir = Path(save_dir)
            adapter_dir.mkdir(parents=True, exist_ok=True)
            peft_model.save_pretrained(str(adapter_dir))
            print(f"Saved PEFT adapter to: {adapter_dir.resolve()}")

        return peft_model

    def run_train(self, adata, **kwargs):
        return self.run_lora_train(adata=adata, **kwargs)
        
    def eval_identity(self, adata):
        pass  

    def eval_perturb(self, adata):
        pass

    def eval_lochness(self, adata):
        pass

    def run_test(self, adata):
        pass
