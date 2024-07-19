import os
import argparse
import torchvision
import numpy as np
import cv2
import torch.nn.functional as F
import matplotlib.pyplot as plt
# from torchprofile import profile_macs
from timm import create_model

from scheduler import create_scheduler
from optim import create_optimizer
import utils 
from loss import create_criterion
from data import create_transform
from data import create_dataset, ImageFolderReader, create_loader
from engine import train_one_epoch, evaluate

import warnings
warnings.filterwarnings("ignore")

import model

import torch

parser = argparse.ArgumentParser(description="pytorch polyp segmetation training")

# Dataset parameters
dataset_group = parser.add_argument_group("Dataset parameters")
dataset_group.add_argument("--data-dir", help="Path to dataset")
dataset_group.add_argument("--dataset", default="folder", help="Dataset name")
dataset_group.add_argument("--save_path", default="res", help="Dataset saving path")
dataset_group.add_argument("--in-chans", type=int, default=3, help="Data in channels")

# Utils parameters
utils_group = parser.add_argument_group("Utils parameters")
utils_group.add_argument('--seed', type=int, default=42, help='Random seed for repudctivity')
utils_group.add_argument('--log-interval', type=int, default=50, help='Frequency of logging output')
utils_group.add_argument('--output-dir', type=str, default="res/", help='path to save checkpoint')

# Model parameters
model_group = parser.add_argument_group("Model parameters")
model_group.add_argument("--model", default="fcbformer", type=str, help="Registry model name")
model_group.add_argument("--pretrained", default=False, action="store_true", help="Use pretrained weight or not")
model_group.add_argument("--num-classes", default=1, type=int, help="Number of classes (default: 1)")
model_group.add_argument("--drop-rate", default=0., type=float, help="Dropout rate")

# Criterion parameters
criterion_group = parser.add_argument_group("Criterion parameters")
criterion_group.add_argument("--criterion", type=str, default="localbcedice", help="Criterion")
criterion_group.add_argument("--smooth", type=float, default=0.7, help="Smooth parameter (default: 0.7)")
criterion_group.add_argument("--alpha", type=float, default=0.5, help="Alpha parameter (default: 0.5)")
criterion_group.add_argument("--beta", type=float, default=0.5, help="Beta parameter (default: 0.5)")
criterion_group.add_argument("--gamma", type=float, default=2, help="Gamma parameter (default: 2)")
criterion_group.add_argument("--ignore-index", type=int, default=-100, help="Ignore index (default: -100)")

# Transform parameters
transform_group = parser.add_argument_group("Transform parameters")
transform_group.add_argument("--img_size", type=int, default=352, help="Image size (default: 352)")
transform_group.add_argument("--crop", type=int, default=0, help="Crop parameter (default: 0)")
transform_group.add_argument("--hflip", type=float, default=0, help="Horizontal flip parameter (default: 0.5)")
transform_group.add_argument("--vflip", type=float, default=0, help="Vertical flip parameter (default: 0.5)")
transform_group.add_argument("--jitter", type=float, default=0, help="Jitter parameter (default: 0.5)")
transform_group.add_argument("--unsharp", type=float, default=0, help="Unsharp parameter (default: 0.5)")
transform_group.add_argument("--cutout", type=int, default=0, help="Cutout parameter (default: 0)")
transform_group.add_argument("--distortion", type=float, default=0, help="Distortion parameter (default: 0.5)")
transform_group.add_argument("--rotate", type=float, default=0, help="Rotate parameter (default: 0.5)")
transform_group.add_argument("--scale", type=float, default=0, help="Scaling parameter (default: 0.5)")
transform_group.add_argument("--affine", type=float, default=0, help="Affine parameter (default: 0.5)")

# Wandb Logger parameters
wandb_group = parser.add_argument_group("W & B parameters")
wandb_group.add_argument('--project', type=str, default="Polyp-Segmentation")
wandb_group.add_argument('--entity', type=str, default="DDCVLAB")
wandb_group.add_argument('--name', type=str, default="MEnet")
wandb_group.add_argument('--wandb', default=False, action="store_true")
wandb_group.add_argument('--logger_train', default=False, action="store_true")

parser.add_argument("--rank", type=int, default=7, help="training device")

parser.add_argument("-b", "--batch-size", default=1, type=int)
parser.add_argument("--amp", default=False, action="store_true",
                    help="Use torch.cuda.amp for mixed precision training")
args, unknown = parser.parse_known_args()

def main(args):

    utils.print_with_timestamp(True)

    device = f"cuda:{args.rank}"
    utils.seed_everything(args.seed)

    model = create_model(args.model)
    model = torch.nn.DataParallel(model, device_ids=[args.rank])
    model.to(device)

    args.name = args.model + "-" + str(args.criterion)
    args.is_master = True

    wandb_logger = None
    if args.wandb and args.is_master:
        wandb_logger = utils.WandbLogger(args)
        wandb_logger.login()
        wandb_logger.init_run()

    args.img_mode = "RGB" if args.in_chans == 3 else "L"

    best_ckpt = torch.load(os.path.join(args.output_dir, "best_ckpt.pth"), map_location=device)
    model.load_state_dict(best_ckpt["model"])

    os.makedirs(os.path.join(args.save_path, args.model), exist_ok=True)


    model.eval()
    for ds in ["Kvasir", "CVC-ClinicDB", "CVC-300", "CVC-ColonDB", "ETIS-LaribPolypDB"]:
        os.makedirs(os.path.join(args.save_path, args.model, ds), exist_ok=True)

        print("*"*5, f" Dataset : {ds}", "*"*5)

        args.dataset = ds
        reader = ImageFolderReader(os.path.join(args.data_dir, args.dataset))

        dataset = create_dataset(
            name=args.dataset,
            root=args.data_dir,
            split=args.dataset,
            reader=reader,
            img_mode=args.img_mode,
        )

        dataset.transform = create_transform(args=args, is_training=False)

        for i in range(len(dataset)):
            images, _ = dataset[i]
            images = images[None].to(device)
            info = dataset.reader[i]
            masks = plt.imread(info["mask_path"])
            masks = np.asarray(masks, np.float32)
            output = model(images)
            output = F.interpolate(output, size=masks.shape, mode='bilinear', align_corners=False)
            output = output.sigmoid().data.cpu().numpy().squeeze()
            output = (output - output.min()) / (output.max() - output.min() + 1e-8)
            cv2.imwrite(os.path.join(args.save_path, args.model, ds, info["name"]), output * 255)
    
    

if __name__ == '__main__':
    main(args)