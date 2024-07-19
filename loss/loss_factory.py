import segmentation_models_pytorch as smp 
from .soft_bce_dice_loss import SoftDiceBCELoss
from .structure_loss import StructureLoss

def criterion_kwargs(args):
    kwargs = dict(
        criterion_name=args.criterion,
        num_classes=getattr(args, "num_classes", 1),

        smooth=getattr(args, "smooth", 0.7),
        alpha=getattr(args, "alpha", 0.5),
        beta=getattr(args, "beta", 0.5),
        gamma=getattr(args, "gamma", 2),

        ignore_index=getattr(args, "ignore_index", -100),
    )
    return kwargs

def create_criterion(
        args,
):
    return _create_criterion(
        **criterion_kwargs(args),
    )

def _create_criterion(
        criterion_name, 
        num_classes,
        smooth,
        alpha,
        beta,
        gamma,
        ignore_index,
        **kwargs,
):

    criterion = None
    if criterion_name == "jaccard":
        criterion = smp.losses.JaccardLoss(
            mode="binary", 
            smooth=smooth
        )
    elif criterion_name == "dice":
        criterion = smp.losses.DiceLoss(
            mode="binary", 
            smooth=smooth
        )
    elif criterion_name == "tversky":
        criterion = smp.losses.TverskyLoss(
            mode="binary", 
            smooth=smooth,
            alpha=alpha,
            beta=beta,
            gamma=gamma
        )
    elif criterion_name == "focal":
        criterion = smp.losses.FocalLoss(
            mode="binary", 
            gamma=gamma
        )
    elif criterion_name == "lovasz":
        criterion = smp.losses.LovaszLoss(
            mode="binary", 
        )
    elif criterion_name == "softbce":
        criterion = smp.losses.SoftBCEWithLogitsLoss(
            ignore_index=ignore_index,
        )
    elif criterion_name == "mcc":
        criterion = smp.losses.MCCLoss()
    elif criterion_name == "bcedice":
        criterion = SoftDiceBCELoss(
            smooth=smooth
        )
    elif criterion_name == "structure":
        criterion = StructureLoss()

    return criterion