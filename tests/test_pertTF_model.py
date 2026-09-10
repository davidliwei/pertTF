import torch

from perttf.model.pertTF import PerturbationTFModel

D_MODEL, N_PERT, N_PS, NTOKENS, N_CLS = 32, 10, 3, 100, 5


def build_model(**overrides):
    """Mirror the constructor call in notebook/train_pertTF_with__lochNESS.ipynb."""
    vocab = {"<pad>": 0, **{f"g{i}": i for i in range(1, NTOKENS)}}
    kwargs = dict(
        vocab=vocab, dropout=0.0, pad_token="<pad>", pad_value=0,
        do_mvc=True, do_dab=False, use_batch_labels=False, num_batch_labels=1,
        domain_spec_batchnorm=False, n_input_bins=0, ecs_threshold=0.7,
        explicit_zero_prob=False, use_fast_transformer=False, pre_norm=False,
        n_cls=N_CLS, nlayers_cls=3,
        pred_lochness_next=True, ps_decoder2_nlayer=5,
    )
    kwargs.update(overrides)
    return PerturbationTFModel(
        N_PERT, 3, N_PS, NTOKENS, D_MODEL, 4, D_MODEL, 2, **kwargs
    ).eval()


def test_ps_decoder2_input_width_matches_pert_encoder_default():
    # Regression for #50: with pert_dim unset, the encoder emits d_model-wide
    # perturbation embeddings, so the decoder must accept 2 * d_model.
    model = build_model()
    first_linear = model.ps_decoder2._decoder[0]
    assert first_linear.in_features == 2 * D_MODEL


def test_forward_with_pred_lochness_next_and_default_pert_dim():
    # The call that crashed in the notebook with
    # "mat1 and mat2 shapes cannot be multiplied (128x64 and 32x32)".
    torch.manual_seed(0)
    model = build_model()
    B, L = 8, 20
    src = torch.randint(1, NTOKENS, (B, L))
    values = torch.rand(B, L)
    mask = torch.zeros(B, L, dtype=torch.bool)
    pert = torch.randint(0, N_PERT, (B,))
    pert_next = torch.randint(0, N_PERT, (B,))

    with torch.no_grad():
        out = model(
            src, values, mask,
            pert_labels=pert, pert_labels_next=pert_next,
            CLS=True, MVC=True, PERTPRED=True, PSPRED=True,
        )

    assert out["ps_output_next"].shape == (B, 1)
    assert torch.isfinite(out["ps_output_next"]).all()
