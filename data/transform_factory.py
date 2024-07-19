import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from albumentations import *
import numpy as np
import timm
import torch.nn.functional as F

STD = timm.data.constants.IMAGENET_DEFAULT_STD
MEAN = timm.data.constants.IMAGENET_DEFAULT_MEAN

def get_mean_and_std(train_data):
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=True)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for X, _ in train_loader:
        for d in range(3):
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
    mean.div_(len(train_data))
    std.div_(len(train_data))
    return list(mean.numpy()), list(std.numpy())

class GridMask(DualTransform):
    """GridMask augmentation for image classification and object detection.
    
    Author: Qishen Ha
    Email: haqishen@gmail.com
    2020/01/29

    Args:
        num_grid (int): number of grid in a row or column.
        fill_value (int, float, lisf of int, list of float): value for dropped pixels.
        rotate ((int, int) or int): range from which a random angle is picked. If rotate is a single int
            an angle is picked from (-rotate, rotate). Default: (-90, 90)
        mode (int):
            0 - cropout a quarter of the square of each grid (left top)
            1 - reserve a quarter of the square of each grid (left top)
            2 - cropout 2 quarter of the square of each grid (left top & right bottom)

    Targets:
        image, mask

    Image types:
        uint8, float32

    Reference:
    |  https://arxiv.org/abs/2001.04086
    |  https://github.com/akuxcw/GridMask
    """

    def __init__(self, num_grid=3, fill_value=0, rotate=0, mode=0, always_apply=False, p=0.5):
        super(GridMask, self).__init__(always_apply, p)
        if isinstance(num_grid, int):
            num_grid = (num_grid, num_grid)
        if isinstance(rotate, int):
            rotate = (-rotate, rotate)
        self.num_grid = num_grid
        self.fill_value = fill_value
        self.rotate = rotate
        self.mode = mode
        self.masks = None
        self.rand_h_max = []
        self.rand_w_max = []

    def init_masks(self, height, width):
        if self.masks is None:
            self.masks = []
            n_masks = self.num_grid[1] - self.num_grid[0] + 1
            for n, n_g in enumerate(range(self.num_grid[0], self.num_grid[1] + 1, 1)):
                grid_h = height / n_g
                grid_w = width / n_g
                this_mask = np.ones((int((n_g + 1) * grid_h), int((n_g + 1) * grid_w))).astype(np.uint8)
                for i in range(n_g + 1):
                    for j in range(n_g + 1):
                        this_mask[
                             int(i * grid_h) : int(i * grid_h + grid_h / 2),
                             int(j * grid_w) : int(j * grid_w + grid_w / 2)
                        ] = self.fill_value
                        if self.mode == 2:
                            this_mask[
                                 int(i * grid_h + grid_h / 2) : int(i * grid_h + grid_h),
                                 int(j * grid_w + grid_w / 2) : int(j * grid_w + grid_w)
                            ] = self.fill_value
                
                if self.mode == 1:
                    this_mask = 1 - this_mask

                self.masks.append(this_mask)
                self.rand_h_max.append(grid_h)
                self.rand_w_max.append(grid_w)

    def apply(self, image, mask, rand_h, rand_w, angle, **params):
        h, w = image.shape[:2]
        mask = F.rotate(mask, angle) if self.rotate[1] > 0 else mask
        mask = mask[:,:,np.newaxis] if image.ndim == 3 else mask
        image *= mask[rand_h:rand_h+h, rand_w:rand_w+w].astype(image.dtype)
        return image

    def get_params_dependent_on_targets(self, params):
        img = params['image']
        height, width = img.shape[:2]
        self.init_masks(height, width)

        mid = np.random.randint(len(self.masks))
        mask = self.masks[mid]
        rand_h = np.random.randint(self.rand_h_max[mid])
        rand_w = np.random.randint(self.rand_w_max[mid])
        angle = np.random.randint(self.rotate[0], self.rotate[1]) if self.rotate[1] > 0 else 0

        return {'mask': mask, 'rand_h': rand_h, 'rand_w': rand_w, 'angle': angle}

    @property
    def targets_as_params(self):
        return ['image']

    def get_transform_init_args_names(self):
        return ('num_grid', 'fill_value', 'rotate', 'mode')

def transform_kwargs(args):
    kwargs = dict(
        img_size=getattr(args, "img_size", 352),

        crop=getattr(args, "crop", 0),
        hflip=getattr(args, "hflip", 0),
        vflip=getattr(args, "vflip", 0),
        jitter=getattr(args, "jitter", 0),
        unsharp=getattr(args, "unsharp", 0),
        cutout=getattr(args, "cutout", 0),
        distortion=getattr(args, "distortion", 0),
        rotate=getattr(args, "rotate", 0),
        scale=getattr(args, "scale", 0),
        affine=getattr(args, "affine", 0),
    )
    return kwargs

def create_transform(
        args,
        is_training,
):
    return _create_transform(
        is_training=is_training,
        **transform_kwargs(args),
    )

def _create_transform(
        img_size,
        crop,
        is_training,
        hflip,
        vflip,
        jitter,
        unsharp,
        cutout,
        distortion,
        rotate,
        scale,
        affine,
        **kwargs,
):

    t = []
    if not is_training:
        t.append(Resize(height=img_size, width=img_size))
    else:
        if crop:
            t.append(A.augmentations.crops.RandomResizedCrop(height=img_size, width=img_size))
        else:
            t.append(Resize(height=img_size, width=img_size))
        if hflip > 0:
            t.append(HorizontalFlip(p=hflip))
        if vflip > 0:
            t.append(VerticalFlip(p=vflip))
        if jitter > 0:
            t.append(ColorJitter(brightness=0.4, contrast=0.5, saturation=0.25, hue=0.01, p=jitter))
        if unsharp > 0:
            t.append(UnsharpMask((25,25), (0.001, 2), p=unsharp))
        if cutout > 0:
            t.append(        
                OneOf([
                    GridMask(num_grid=3, mode=0),
                    GridMask(num_grid=3, mode=1),
                    GridMask(num_grid=3, mode=2),
                    CoarseDropout(max_holes=8, max_height=32, max_width=32, fill_value=0, p=cutout),
                ], p=cutout),
            )
        # if scale > 0:
        #     t.append(RandomScale(scale_limit=0.5, p=scale))
        if affine > 0:
            t.append(Affine(shear=22.5, rotate=90, translate_px=48, p=affine))
        if distortion > 0:
            t.append(GridDistortion(p=distortion))
        # if rotate > 0:
        #     t.append(Rotate(limit=48, p=rotate))
        
    t.append(Normalize(mean=MEAN, std=STD))
    t.append(ToTensorV2())
    

    return Compose(t)