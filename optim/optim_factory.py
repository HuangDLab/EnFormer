from torch.optim import SGD, Adam, AdamW
from .lion import Lion

def optimizer_kwargs(args):

    kwargs = dict(
        opt=args.opt,
        lr=getattr(args, "lr", 1e-4),
        wd=getattr(args, "wd", 2e-5),
        momentum=getattr(args, "momentum", 0.9),
        beta1=getattr(args, "beta1", 0.9),
        beta2=getattr(args, "beta2", 0.999)
    )

    return kwargs

def create_optimizer(params_to_optim, args):
    return _create_optimizer(
        params_to_optim,
        **optimizer_kwargs(args)
    )

def _create_optimizer(
        params_to_optim,
        opt,
        lr,
        wd,
        momentum,
        beta1,
        beta2,
        **kwargs
):
    opt = opt.lower()
    opt_base_args = dict(
        lr=lr,
        weight_decay=wd,
    )

    optimizer = None
    if opt == "sgd":
        optimizer = SGD(
            params=params_to_optim,
            **opt_base_args,
            momentum=momentum
        )
    elif opt == "adam":
        optimizer = Adam(
            params=params_to_optim,
            **opt_base_args,
            betas=(beta1, beta2)
        )
    elif opt == "adamw":
        optimizer = AdamW(
            params_to_optim,
            **opt_base_args,
            betas=(beta1, beta2),
        )
    elif opt == "lion":
        optimizer = Lion(
            params_to_optim,
            **opt_base_args,
            betas=(beta1, beta2)
        )
    return optimizer
    