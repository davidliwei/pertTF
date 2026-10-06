import os
import json
import hashlib
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
        # to_container: list values (e.g. special_tokens) are OmegaConf ListConfig, which json cannot write.
        final_train_config = OmegaConf.to_container(OmegaConf.create(self.training_config), resolve=True) if self.training_config else {}
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
        # Fingerprint of the checkpoint weights, taken before loading (which renames attention keys per
        # backend); LoRA adapters record it so load_lora_adapter can check they get the same base.
        weights_hash = hashlib.sha256()
        for key in sorted(state_dict):
            weights_hash.update(key.encode())
            weights_hash.update(state_dict[key].contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        base_hash = weights_hash.hexdigest()

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
        model.base_hash = base_hash

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

    @classmethod
    def from_adata(cls, adata, config: Dict):
        """
        New untrained model for run_train. The gene vocabulary and the cell type / genotype label dictionaries
        come from adata (var names, obs 'celltype' and obs 'genotype'), built as the training data loader builds
        them; architecture and default training settings come from config (the keys of a pertTF training config,
        e.g. layer_size, nlayers, nhead, GEPC, distribution, sampling_mode, lr, epochs).
        """
        special_tokens = config.get("special_tokens", ["<pad>", "<cls>", "<eoc>"])
        vocab = SimpleVocab(adata.var.index.tolist(), special_tokens)
        vocab.set_default_index(vocab["<pad>"])
        genotype_to_index = {g: i for i, g in enumerate(adata.obs["genotype"].unique())}
        cell_type_to_index = {c: i for i, c in enumerate(adata.obs["celltype"].unique())}
        return cls(vocab=vocab, genotype_to_index=genotype_to_index, cell_type_to_index=cell_type_to_index, **config)

    def run_train(
        self,
        adata,
        train_config: Optional[Dict] = None,
        pretrain: bool = False,
        train_indices=None,
        valid_indices=None,
        input_layer_key: str = "X_binned",
        save_dir: Optional[str] = None,
        device: Optional[torch.device] = None,
        wandb_mode: str = "disabled",
    ):
        """
        Full training of all weights with train_function.wrapper_train, for a new model from from_adata or a
        loaded checkpoint (continued training or full fine-tuning). The settings are this model's
        training_config updated with train_config (e.g. {'epochs': 5, 'lr': 1e-4}); the objective follows
        next_cell_pred_type ('identity', 'pert' or 'lochness') as in wrapper_train. 'pert' needs explicit
        train_indices / valid_indices; otherwise a random split is made when they are omitted.

        pretrain=True: label-free pretraining (masked-gene prediction and reconstruction of each cell's own
        expression), overriding any conflicting setting: identity objective, next_weight=0, this_weight=1,
        cell type and genotype classifiers off, CCE off, perturbation_input off, PS/lochNESS heads off.
        adata.obs still needs 'celltype' and 'genotype' columns (any values) for the data loader.

        The best epoch is restored into this model, the resolved settings are stored in training_config, and
        with save_dir the model is saved there with save_pretrained (wrapper_train's own per-epoch files go to
        save_dir/wrapper_train). wandb_mode: 'disabled' (default), 'offline' or 'online'. Returns self.
        """
        import tempfile
        from . import lora, train_function
        from .config_gen import generate_config
        from .train_data_gen import produce_training_datasets

        settings = OmegaConf.to_container(self._init_default_train_config_(), resolve=True)
        settings.update(train_config or {})
        # Avoid duplicate kwarg collision in PertBatchCollator(vocab=..., **config).
        settings.pop("vocab", None)
        if pretrain:
            settings.update(next_cell_pred_type="identity", next_weight=0, this_weight=1.0, cell_type_classifier=False,
                            genotype_classifier=False, CCE=False, perturbation_input=False, ps_weight=0.0)
            settings["mask_ratio"] = settings["mask_ratio"] if settings["mask_ratio"] > 0 else 0.15
            # train() enables the next-cell lochNESS loss whenever this key exists, whatever its value.
            settings.pop("pred_lochness_next", None)
        # Loss flags follow the modules this model actually has.
        settings["GEPC"] = hasattr(self, "mvc_decoder")
        settings["explicit_zero_prob"] = bool(self.explicit_zero_prob)
        settings["distribution"] = self.distribution
        config, run = generate_config(settings, wandb_mode=wandb_mode)

        # Classifier heads and the perturbation encoder index the model's own label dictionaries.
        require_known = []
        if config.cell_type_classifier:
            require_known.append("celltype")
        if config.genotype_classifier or config.next_cell_pred_type != "identity":
            require_known.append("genotype")
        cell_type_to_index, genotype_to_index = lora.label_mappings(self, adata, require_known=require_known)
        ps_columns = [c for c in (getattr(self, "ps_names", None) or []) if c in adata.obs.columns] or None
        data_gen = produce_training_datasets(
            adata_input=adata,
            config=config,
            input_layer_key=input_layer_key,
            next_cell_pred=config.next_cell_pred_type,
            cell_type_to_index=cell_type_to_index,
            genotype_to_index=genotype_to_index,
            vocab=self.vocab,
            ps_columns=ps_columns,
            train_indices=train_indices,
            valid_indices=valid_indices,
        )

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        with tempfile.TemporaryDirectory() as tmp_dir:
            work_dir = Path(save_dir) / "wrapper_train" if save_dir else Path(tmp_dir)
            work_dir.mkdir(parents=True, exist_ok=True)
            best_model = train_function.wrapper_train(self, config, data_gen, save_dir=work_dir, device=device)
        self.load_state_dict(best_model.state_dict())
        self.training_config = OmegaConf.create({k: v for k, v in config.as_dict().items() if not k.startswith("_")})
        if run is not None:
            run.finish()
        if save_dir:
            self.save_pretrained(save_dir)
        return self

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
        
    def run_lora_cls_train(
        self,
        adata,
        cls_col: str,
        cls_task: str,
        epochs: int = 1,
        batch_size: Optional[int] = None,
        lr: Optional[float] = None,
        train_val_split: float = 0.2,
        input_layer_key: str = "X_binned",
        lora_config=None,
        device: Optional[torch.device] = None,
        save_dir: Optional[str] = None,
        amp: Optional[bool] = None,
        seed: Optional[int] = None,
        train_indices=None,
        valid_indices=None,
        stratify_by=None,
        split_check_columns=None,
        this_weight: float = 1.0,
    ):
        """
        LoRA fine-tuning with a new head that predicts a cell-level target from the cell embedding.

        cls_col: adata.obs column to predict, with cls_task 'classification' or 'regression'. The head
            is trained from scratch on the L2-normalised <cls> embedding (obsm['X_scGPT'] after
            eval_testdata) and saved in the adapter together with its labels.
        this_weight: weight of the expression reconstruction losses of the cell itself (masked-gene MSE
            and GEPC/MVC), trained in the same forward pass on masked input; 0 trains the head loss only
            on unmasked input.
        lr: AdamW learning rate (default: the checkpoint's training_config lr), multiplied by schedule_ratio
            after each epoch; the checkpoint's optimizer/scheduler settings are not used.
        The split is random (train_val_split, optionally stratify_by) unless train_indices/valid_indices
        are given. For classification, cls_col is checked so every validation class is also in training;
        split_check_columns adds further columns to check. Checkpoints are selected on the validation loss of the head.
        """
        import copy
        import numpy as np
        from . import lora
        from .modules import ClsDecoder
        from .train_data_gen import produce_training_datasets

        if cls_task not in ("classification", "regression"):
            raise ValueError(f"cls_task must be 'classification' or 'regression'; got {cls_task!r}")
        lora.check_no_adapter(self)

        config, ps_columns = lora.prepare_config(self, adata, input_layer_key, batch_size, lr, amp, None, seed)
        # Always the identity objective, whatever the checkpoint was trained for (e.g. a pert checkpoint
        # has this_weight=0, next_weight=1, mask_ratio=0); the pretrained classifier heads are not trained.
        config.next_cell_pred_type = "identity"
        config.next_weight = 0.0
        config.this_weight = this_weight
        config.cell_type_classifier = False
        config.genotype_classifier = False
        # Reconstruction needs masked genes; the head alone uses unmasked input.
        if this_weight > 0:
            config.mask_ratio = config.mask_ratio if config.mask_ratio > 0 else 0.15
        else:
            config.mask_ratio = 0.0

        # The genotype and celltype mappings only feed the data loader here (their heads are not trained).
        cell_type_to_index, genotype_to_index = lora.label_mappings(self, adata, require_known=[])

        if cls_task == "classification":
            cls_to_index = {str(x): i for i, x in enumerate(sorted(adata.obs[cls_col].astype(str).unique()))}
            cls_targets = torch.tensor(adata.obs[cls_col].astype(str).map(cls_to_index).to_numpy(), dtype=torch.long)
            n_out = len(cls_to_index)
        else:
            cls_to_index = None
            cls_targets = torch.tensor(adata.obs[cls_col].to_numpy(dtype=np.float32))
            n_out = 1
        cls_info = {"cls_col": cls_col, "cls_task": cls_task, "cls_to_index": cls_to_index, "n_out": n_out}
        self.cls_head = ClsDecoder(self.d_model, n_out)
        self.cls_info = cls_info
        lora_config = copy.deepcopy(lora_config) if lora_config is not None else self.build_lora_config()
        # The new head is trained fully and saved inside the adapter.
        lora_config.modules_to_save = list(lora_config.modules_to_save or []) + ["cls_head"]

        # Only the cls_col labels are learned, so by default only they are checked (validation classes
        # must be in training); continuous regression targets are not checked.
        if split_check_columns is None:
            split_check_columns = []
        elif isinstance(split_check_columns, str):
            split_check_columns = [split_check_columns]
        if cls_task == "classification" and cls_col not in split_check_columns:
            split_check_columns = list(split_check_columns) + [cls_col]

        data_gen = produce_training_datasets(
            adata_input=adata,
            config=config,
            input_layer_key=input_layer_key,
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
        peft_model, optimizer_dict = lora.wrap(self, lora_config, config, device, data_gen["num_batch_types"])

        epoch_kwargs = dict(config=config, device=device, pad_id=data_gen["vocab"][config.pad_token],
                            cls_targets=cls_targets, cls_task=cls_task, optimizer_dict=optimizer_dict)

        def run_epoch(epoch):
            train_head, train_recon = lora.cls_epoch(peft_model, self.cls_head, data_gen["train_loader"], is_train=True, **epoch_kwargs)
            val_head, val_recon = lora.cls_epoch(peft_model, self.cls_head, data_gen["valid_loader"], is_train=False, **epoch_kwargs)
            print(f"Epoch {epoch}/{epochs} | train head: {train_head:.4f} recon: {train_recon:.4f} | val head: {val_head:.4f} recon: {val_recon:.4f}")
            return val_head

        lora.fit(peft_model, epochs, run_epoch, device, optimizer_dict["scheduler"])
        lora.save(peft_model, save_dir, {"mode": "cls", "cls_info": cls_info})
        return peft_model

    def run_lora_pert_train(
        self,
        adata,
        train_indices,
        valid_indices,
        epochs: int = 1,
        batch_size: Optional[int] = None,
        lr: Optional[float] = None,
        input_layer_key: str = "X_binned",
        lora_config=None,
        device: Optional[torch.device] = None,
        save_dir: Optional[str] = None,
        amp: Optional[bool] = None,
        log_interval: Optional[int] = None,
        seed: Optional[int] = None,
        checkpoint_metric: str = "pearson_delta",
        pert_mode: str = "ft",
        pert_source: str = "denovo",
        distribution: Optional[str] = "nb",
    ):
        """
        LoRA fine-tuning of perturbation prediction (control cell + target perturbation -> perturbed expression).

        train_indices / valid_indices are required because the split defines what is evaluated: validation
        perturbations or cell types absent from training measure transfer to unseen perturbations or contexts.
        Each split needs control ('WT') and perturbed cells, and every validation cell type needs both.

        pert_mode:
            'ft': a checkpoint trained with next_cell_pred_type='pert'; keeps its objective, perturbation labels
                (adata genotypes must all be model labels) and expression decoder. LoRA also adapts the
                perturbation encoder and the perturbed-cell encoder (pert_exp_encoder).
            'new': any checkpoint, including one pretrained only on unperturbed cells (identity). A new
                perturbation encoder, perturbed-cell encoder and expression decoder are trained from scratch
                and saved with the adapter, together with the new perturbation dictionary (adata genotypes,
                plus every gene of the feature sources). The objective is the perturbed expression plus the
                reconstruction of the cell itself (next_weight = this_weight = 1); the pretrained classifier
                and PS heads are not trained.
        pert_source ('new' only): perturbation embedding, 'denovo' (one learned vector per perturbation) or
            feature presets 'esm2', 'genept', 'gears' (joinable with '+', e.g. 'esm2+genept'). Only feature
            sources can embed perturbations absent from training, so held-out validation perturbations must
            be covered by the source.
        distribution ('new' only): expression distribution of the new decoder, 'nb' (default), 'zinb', 'hnb',
            'zig', 'pois', 'zipois', or None (MSE).

        checkpoint_metric selects the best epoch: 'pearson_delta' (default), 'mse_delta',
        'ttest_de_overlap_at_n', 'ttest_de_direction_match' or 'mvc_next' (validation loss of the predicted
        perturbed expression). The delta metrics compare, per (cell type, perturbation) group with at least
        30 cells, the mean change from control of sampled predictions and of the observed input_layer_key
        expression, over all genes (no precomputed DE genes).

        lr: AdamW learning rate (default: the checkpoint's training_config lr), multiplied by schedule_ratio
        after each epoch; the checkpoint's optimizer/scheduler settings are not used.
        """
        import copy
        import numpy as np
        from torch import nn
        from . import lora
        from . import train_function
        from .train_data_gen import produce_training_datasets
        from ..utils.pert_metrics import compute_perturbation_metrics, group_moments_from_anndata, resolve_checkpoint_score

        if pert_mode not in ("ft", "new"):
            raise ValueError(f"pert_mode must be 'ft' or 'new'; got {pert_mode!r}")
        lora.check_no_adapter(self)
        checkpoint_mode = (self.training_config or {}).get("next_cell_pred_type")
        if pert_mode == "ft" and checkpoint_mode != "pert":
            raise ValueError(
                f"pert_mode='ft' needs a checkpoint trained for perturbation prediction "
                f"(training_config.next_cell_pred_type='pert'); this one has {checkpoint_mode!r}. "
                "Use pert_mode='new' to train new perturbation modules on it."
            )

        config, ps_columns = lora.prepare_config(self, adata, input_layer_key, batch_size, lr, amp, log_interval, seed)
        config.next_cell_pred_type = "pert"
        lora_config = copy.deepcopy(lora_config) if lora_config is not None else self.build_lora_config()

        if pert_mode == "ft":
            # Keep the checkpoint's perturbation objective (next_weight, this_weight, mask_ratio, classifier flags).
            # LoRA also adapts the pretrained perturbation encoder and perturbed-cell encoder.
            pert_layers = [
                f"{prefix}.{name}"
                for prefix in ("pert_encoder", "pert_exp_encoder")
                for name, module in getattr(self, prefix).named_modules()
                if isinstance(module, (nn.Linear, nn.Embedding))
            ]
            lora_config.target_modules = list(lora_config.target_modules) + pert_layers
            pert_info = None
        else:
            from .pert_emb import build_perturbation_mapping, load_perturbation_sources

            pert_sources = load_perturbation_sources(None if pert_source == "denovo" else pert_source)
            new_genotype_to_index = build_perturbation_mapping(sorted(adata.obs["genotype"].astype(str).unique()), pert_sources)
            lora.attach_new_pert_modules(self, new_genotype_to_index, pert_sources, distribution)
            # Validation perturbations absent from training are only meaningful with features from a source;
            # otherwise their vector in the new encoder is never trained.
            genotype = adata.obs["genotype"].astype(str)
            unseen = set(genotype.iloc[valid_indices]) - set(genotype.iloc[train_indices])
            untrained = sorted(g for g in unseen if self.pert_encoder.fallback_indices[new_genotype_to_index[g]] >= 0)
            if untrained:
                raise ValueError(
                    f"Validation perturbations {untrained} are not in training and have no features in "
                    f"pert_source={pert_source!r}, so their embedding would stay untrained. Use a feature source "
                    "that covers them, or validate on perturbations that are also in training."
                )
            # New modules are trained fully and saved inside the adapter.
            lora_config.modules_to_save = list(lora_config.modules_to_save or []) + ["pert_encoder", "pert_exp_encoder", "mvc_decoder"]
            pert_info = {
                "genotype_to_index": new_genotype_to_index,
                "sources": {
                    name: {"genes": source["genes"], "dim": int(self.pert_encoder.sources[name].features.shape[1])}
                    for name, source in self.pert_encoder.source_config()["sources"].items()
                },
                "distribution": distribution,
            }
            config.GEPC = True
            config.distribution = distribution
            config.next_weight = 1.0
            config.this_weight = 1.0
            config.mask_ratio = config.mask_ratio if config.mask_ratio > 0 else 0.15
            config.cell_type_classifier = False
            config.genotype_classifier = False
            config.ps_weight = 0.0

        # In 'ft' mode genotype labels select rows of the pretrained perturbation embedding, so they must all be
        # model labels; in 'new' mode the model already holds the new dictionary.
        cell_type_to_index, genotype_to_index = lora.label_mappings(self, adata, require_known=["genotype"])

        data_gen = produce_training_datasets(
            adata_input=adata,
            config=config,
            input_layer_key=input_layer_key,
            next_cell_pred="pert",
            cell_type_to_index=cell_type_to_index,
            genotype_to_index=genotype_to_index,
            vocab=self.vocab,
            ps_columns=ps_columns,
            train_indices=train_indices,
            valid_indices=valid_indices,
        )

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        peft_model, optimizer_dict = lora.wrap(self, lora_config, config, device, data_gen["num_batch_types"])

        # Observed group means of the validation cells, from the data manager's AnnData so the genes are in
        # the same order as the predicted groups.
        target_sum = config.get("perturbation_metric_target_sum", 10000.0)
        reference_indices = np.concatenate([data_gen["pert_valid_target_indices"], data_gen["pert_valid_control_indices"]])
        real_groups = group_moments_from_anndata(
            data_gen["adata_manager"].adata, reference_indices, layer=input_layer_key, input_scale="log1p", target_sum=target_sum,
        ).finalize()
        metric_name = checkpoint_metric if checkpoint_metric == "mvc_next" else f"{checkpoint_metric}_sample"

        def run_epoch(epoch):
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
            evaluation = train_function.evaluate(
                model=peft_model,
                loader=data_gen["valid_loader"],
                config=config,
                vocab=data_gen["vocab"],
                cell_type_to_index=data_gen["cell_type_to_index"],
                genotype_to_index=data_gen["genotype_to_index"],
                expr_prediction_mode="sample",
                target_sum=target_sum,
                sample_seed=config.get("seed", None),
                epoch=epoch,
                device=device,
            )
            aggregate = compute_perturbation_metrics(
                real_groups, evaluation["predicted_groups"], control_value="WT"
            )["aggregate"]
            validation_metrics = {"mvc_next": evaluation["losses"]["mvc_next"]}
            validation_metrics.update({f"{name}_sample": value for name, value in aggregate.items()})
            print(
                f"Epoch {epoch}/{epochs} | val pearson_delta: {aggregate['pearson_delta']:.4f} | "
                f"mse_delta: {aggregate['mse_delta']:.4f} | mvc_next: {validation_metrics['mvc_next']:.4f} | "
                f"groups: {aggregate['n_evaluated_groups']}"
            )
            score, _, mode = resolve_checkpoint_score(validation_metrics, metric_name)
            # lora.fit keeps the lowest score
            return score if mode == "min" else -score

        lora.fit(peft_model, epochs, run_epoch, device, optimizer_dict["scheduler"])
        lora.save(peft_model, save_dir, {"mode": "pert", "cls_info": None, "pert_info": pert_info})
        return peft_model

    @classmethod
    def load_lora_adapter(cls, base_model_name_or_path: str, adapter_dir: str, **kwargs):
        """
        Load the base checkpoint (HF repo id or local dir; kwargs go to from_pretrained) and attach an
        adapter saved by run_lora_cls_train or run_lora_pert_train, rebuilding the cls head or the
        pert_mode='new' perturbation modules (with their perturbation dictionary and distribution) first when
        they were trained. Each call returns an independent model, so several adapters can be loaded side by side.
        """
        from peft import PeftModel
        from . import lora
        from .modules import ClsDecoder

        with open(Path(adapter_dir) / "lora_heads.json") as f:
            heads = json.load(f)
        model = cls.from_pretrained(base_model_name_or_path, **kwargs)
        # A different checkpoint with the same architecture would accept the adapter silently.
        if model.base_hash != heads["base_hash"]:
            raise ValueError(
                f"{base_model_name_or_path} is not the base model the adapter in {adapter_dir} was trained on "
                "(checkpoint weights differ)."
            )
        if heads["cls_info"] is not None:
            model.cls_info = heads["cls_info"]
            model.cls_head = ClsDecoder(model.d_model, model.cls_info["n_out"])
        pert_info = heads.get("pert_info")
        if pert_info is not None:
            # Feature tables of the saved shape; their values are restored from the adapter with the trained weights.
            pert_sources = {name: (source["genes"], torch.zeros(len(source["genes"]), source["dim"]))
                            for name, source in pert_info["sources"].items()}
            lora.attach_new_pert_modules(model, pert_info["genotype_to_index"], pert_sources, pert_info["distribution"])
        return PeftModel.from_pretrained(model, adapter_dir)

    def predict_cls(self, adata, input_layer_key: str = "X_binned"):
        """
        Predict the trained cls_col for each cell: embeds the cells with this LoRA-adapted model (the
        L2-normalised <cls> embedding the head was trained on), then applies the cls head. Call it on the
        model returned by run_lora_cls_train or load_lora_adapter. Returns a pandas Series indexed by cell name.
        """
        import numpy as np
        import pandas as pd
        from .train_function import eval_testdata

        device = next(self.parameters()).device
        # The head was trained in identity mode, whatever the checkpoint's own objective.
        config = OmegaConf.merge(self.training_config, {"next_cell_pred_type": "identity"})
        train_data_dict = {"genotype_to_index": self.genotype_to_index, "vocab": self.vocab,
                           "cell_type_to_index": self.cell_type_to_index}
        adata_eval = eval_testdata(self, adata, None, train_data_dict=train_data_dict, config=config,
                                   input_layer_key=input_layer_key, device=device)
        with torch.no_grad():
            pred = self.cls_head(torch.as_tensor(adata_eval.obsm["X_scGPT"], dtype=torch.float32, device=device)).cpu().numpy()
        if self.cls_info["cls_task"] == "regression":
            values = pred[:, 0]
        else:
            index_to_label = {i: label for label, i in self.cls_info["cls_to_index"].items()}
            values = np.array([index_to_label[i] for i in pred.argmax(axis=1)])
        return pd.Series(values, index=adata_eval.obs_names, name=f"predicted_{self.cls_info['cls_col']}")

    def eval_identity(self, adata):
        pass  

    def eval_perturb(self, adata):
        pass

    def eval_lochness(self, adata):
        pass

    def run_test(self, adata):
        pass
