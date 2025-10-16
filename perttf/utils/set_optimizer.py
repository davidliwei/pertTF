import torch
from scgpt.model import TransformerModel, AdversarialDiscriminator

def create_optimizer_dict(model, device, config, num_batch_types = -1):
    scaler = torch.cuda.amp.GradScaler(enabled=config.amp)
    DAB_separate_optim = True if config.dab_weight >0 else False

    # This maybe should be part of training code 
    if config.ADV and num_batch_types > 1:
        discriminator = AdversarialDiscriminator(
            d_model=config.layer_size, # embsize
            n_cls=num_batch_types,
        ).to(device)
        print(discriminator)
    else:
        discriminator = None

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, eps=1e-4 if config.amp else 1e-8
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=config.schedule_ratio)

    if DAB_separate_optim:
        optimizer_dab = torch.optim.Adam(model.parameters(), lr=config.lr)
        scheduler_dab = torch.optim.lr_scheduler.StepLR(
            optimizer_dab, config.schedule_interval, gamma=config.schedule_ratio
        )
    else:
        optimizer_dab = None
        scheduler_dab = None

    if config.ADV:
        optimizer_E = torch.optim.Adam(model.parameters(), lr=config.lr_ADV)
        scheduler_E = torch.optim.lr_scheduler.StepLR(
            optimizer_E, config.schedule_interval, gamma=config.schedule_ratio
        )
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=config.lr_ADV)
        scheduler_D = torch.optim.lr_scheduler.StepLR(
            optimizer_D, config.schedule_interval, gamma=config.schedule_ratio
        )
    else:
        optimizer_E = None
        scheduler_E = None
        optimizer_D = None
        scheduler_D = None

    optimizer_dict={
        "scaler": scaler,
        "discriminator": discriminator,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "optimizer_dab": optimizer_dab,
        "scheduler_dab": scheduler_dab,
        "optimizer_E": optimizer_E,
        "scheduler_E": scheduler_E,
        "optimizer_D": optimizer_D,
        "scheduler_D": scheduler_D,
        'DAB_separate_optim': DAB_separate_optim
    }
    return optimizer_dict
