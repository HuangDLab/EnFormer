from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, OneCycleLR
from .poly_lr_scheduler import poly_lr_scheduler

def scheduler_kwargs(args):
    kwargs = dict(
        scheduler_name=args.scheduler_name,
        num_epochs=getattr(args, "epochs", 100),

        min_lr=getattr(args, "min_lr", 1e-6),
        max_lr=getattr(args, "max_lr", args.lr),

        warmup_lr=getattr(args, "warmup_lr", 1e-5),
        warmup_pct=getattr(args, "warmup-pct", 0.3),
        warmup_epochs=getattr(args, "warmup_epochs", 5),
    )
    return kwargs

def create_scheduler(
        optimizer, 
        args,
        steps_per_epoch,
        step_on_epochs,
):
    return _create_scheduler(
        optimizer=optimizer,
        steps_per_epoch=steps_per_epoch,
        step_on_epochs=step_on_epochs,
        **scheduler_kwargs(args),
    )

def _create_scheduler(
        optimizer,
        scheduler_name, 
        num_epochs,
        max_lr,
        min_lr,
        warmup_epochs,
        warmup_pct,
        steps_per_epoch,
        step_on_epochs,
        **kwargs,
):
    warmup_steps = warmup_epochs
    maximum_steps = num_epochs
    if not step_on_epochs:
        maximum_steps = maximum_steps * steps_per_epoch
        warmup_steps = warmup_steps * steps_per_epoch

    warmup_args = dict(
        eta_min=min_lr,
        T_0=warmup_steps,
    )

    consine_args = dict(
        T_max=maximum_steps,
        eta_min=min_lr,
    )

    cycle_args = dict(
        epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=warmup_pct,
        max_lr=max_lr
    )

    poly_args = dict(
        num_steps=steps_per_epoch,
        epochs=num_epochs,
    )

    scheduler = None
    if scheduler_name == "warmup":
        scheduler = CosineAnnealingWarmRestarts(optimizer,**warmup_args)
    elif scheduler_name == "consine":
        scheduler = CosineAnnealingLR(optimizer, **consine_args)
    elif scheduler_name == "cycle":
        scheduler = OneCycleLR(optimizer, **cycle_args)
    elif scheduler_name == "poly":
        scheduler = poly_lr_scheduler(optimizer, **poly_args)

    return scheduler