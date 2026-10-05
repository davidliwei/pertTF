import math
from typing import Optional

import torch
from ..model.modules import AdversarialDiscriminator

def create_optimizer_dict(model, device, config, num_batch_types = -1, *, steps_per_epoch: Optional[int] = None):
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp)
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

    optimizer_name = config.get('optimizer', 'adam').lower()
    eps = 1e-4 if config.amp else 1e-8
    optimizer_config = {'optimizer': optimizer_name, 'lr': config.lr}
    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, eps=eps)
    elif optimizer_name == 'adamw':
        weight_decay = config.get('weight_decay', 0.01)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, eps=eps,
                                      weight_decay=weight_decay)
        optimizer_config['weight_decay'] = weight_decay
    elif optimizer_name == 'muon':
        from muon import MuonWithAuxAdam, SingleDeviceMuonWithAuxAdam

        # Embedding names are not reliable (e.g. encoder.embedding vs encoder.table).
        embedding_ids = {
            id(parameter)
            for module in model.modules()
            if isinstance(module, (torch.nn.Embedding, torch.nn.EmbeddingBag))
            for parameter in module.parameters(recurse=False)
        }
        muon_params, adam_params = [], []
        for parameter in model.parameters():
            if not parameter.requires_grad:
                continue
            if parameter.ndim >= 2 and id(parameter) not in embedding_ids:
                muon_params.append(parameter)
            else:
                adam_params.append(parameter)
        aux_lr_ratio = config.get('muon_aux_lr_ratio', 0.01)
        weight_decay = config.get('weight_decay', 0.01)
        param_groups = [
            {'params': muon_params, 'use_muon': True, 'lr': config.lr,
             'momentum': 0.95, 'weight_decay': weight_decay},
            {'params': adam_params, 'use_muon': False, 'lr': config.lr * aux_lr_ratio,
             'betas': (0.9, 0.95), 'weight_decay': weight_decay},
        ]
        muon_class = MuonWithAuxAdam if torch.distributed.is_initialized() else SingleDeviceMuonWithAuxAdam
        optimizer = muon_class(param_groups)
        optimizer_config.update(muon_aux_lr_ratio=aux_lr_ratio, weight_decay=weight_decay)
    else:
        raise ValueError(f'Unknown optimizer: {optimizer_name}')

    scheduler_name = config.get('scheduler', 'step')
    warmup_epochs = config.get('warmup_epochs', 0.0)
    if scheduler_name == 'step':
        if warmup_epochs != 0:
            raise ValueError('warmup_epochs requires scheduler="cosine"')
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=config.schedule_ratio)
        scheduler_interval = 'epoch'
    elif scheduler_name == 'cosine':
        if not 0 <= warmup_epochs < config.epochs:
            raise ValueError('Cosine scheduling requires 0 <= warmup_epochs < epochs')
        if steps_per_epoch is None:
            raise ValueError('Cosine scheduling requires steps_per_epoch=len(train_loader)')
        total_updates = config.epochs * steps_per_epoch
        warmup_updates = int(warmup_epochs * steps_per_epoch)

        def lr_multiplier(update: int) -> float:
            if update < warmup_updates:
                return 0.01 + 0.99 * update / warmup_updates
            progress = (update - warmup_updates) / (total_updates - warmup_updates)
            return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
        scheduler_interval = 'update'
        optimizer_config.update(updates_per_epoch=steps_per_epoch, total_updates=total_updates,
                                warmup_updates=warmup_updates)
    else:
        raise ValueError(f'Unknown scheduler: {scheduler_name}')
    optimizer_config.update(scheduler=scheduler_name, scheduler_interval=scheduler_interval,
                            warmup_epochs=warmup_epochs)

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
        "scheduler_interval": scheduler_interval,
        "optimizer_config": optimizer_config,
        "optimizer_dab": optimizer_dab,
        "scheduler_dab": scheduler_dab,
        "optimizer_E": optimizer_E,
        "scheduler_E": scheduler_E,
        "optimizer_D": optimizer_D,
        "scheduler_D": scheduler_D,
        'DAB_separate_optim': DAB_separate_optim
    }
    return optimizer_dict
