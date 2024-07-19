import torch
import torchvision.transforms as T
from .transform_factory import _create_transform, transform_kwargs

def loader_kwargs(args):
    kwargs = dict(
        batch_size=getattr(args, "batch_size", 4),
        num_workers=getattr(args, "num_workers", 0),
        pin_memory=getattr(args, "pin_memeory", True),
    )
    return kwargs

def create_loader(
        dataset,
        args,
        **kwargs
):
    return _create_loader(
        dataset,
        **loader_kwargs(args),
        **transform_kwargs(args),
        **kwargs
    )

def _create_loader(
        dataset,
        batch_size,
        num_workers, 
        pin_memory,
        is_training, 

        img_size,  
        crop,
        hflip, 
        vflip, 
        jitter,
        unsharp,
        cutout,
        distortion,
        rotate,
        scale,
        affine,
):
    dataset.transform = _create_transform(
        is_training=is_training,
        img_size=img_size,
        crop=crop,  
        hflip=hflip, 
        vflip=vflip, 
        jitter=jitter,
        unsharp=unsharp,
        cutout=cutout,
        distortion=distortion,
        rotate=rotate,
        scale=scale,
        affine=affine,
    )

    batch_size = batch_size if is_training else 1

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=is_training,
        shuffle=is_training
    )
    return loader