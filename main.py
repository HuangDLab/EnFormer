import os
import argparse
from timm import create_model

from scheduler import create_scheduler
from optim import create_optimizer
import utils 
from loss import create_criterion
from data import create_dataset, ImageFolderReader, create_loader
from engine import train_one_epoch, evaluate
import torch
import warnings

warnings.filterwarnings("ignore")

import model


parser = argparse.ArgumentParser(description="pytorch polyp segmetation training")

# Dataset parameters
dataset_group = parser.add_argument_group("Dataset parameters")
dataset_group.add_argument("--data-dir", help="Path to dataset")
dataset_group.add_argument("--dataset", default="folder", help="Dataset name")
dataset_group.add_argument("--train-split", default="train", help="Dataset train split")
dataset_group.add_argument("--val-split", default="val", help="Dataset validation split")
dataset_group.add_argument("--test-split", default="test", help="Dataset test split")
dataset_group.add_argument("--in-chans", type=int, default=3, help="Data in channels")

# Scheduler parameters
scheduler_group = parser.add_argument_group("Scheduler parameters")
scheduler_group.add_argument("--scheduler-name", default="cycle", type=str, help="Scheduler name")
scheduler_group.add_argument("--epochs", default=200, type=int, help="Number of epochs")

scheduler_group.add_argument("--min-lr", default=1e-6, type=float, help="Minimum learning rate for scheduler")
scheduler_group.add_argument("--max-lr", default=1e-4, type=float, help="Maximum learning rate for scheduler")

scheduler_group.add_argument("--warmup-lr", default=1e-5, type=float, help="Warm-up learning rate for scheduler")
scheduler_group.add_argument("--warmup-pct", default=0.3, type=float, help="Warm-up percentage for scheduler")
scheduler_group.add_argument("--warmup-epochs", default=5, type=int, help="Warm-up epochs for scheduler")

# Optimizer parameters
optimizer_group = parser.add_argument_group("Optimizer parameters")
optimizer_group.add_argument("--opt", default="adam", type=str, help="Optimizer")
optimizer_group.add_argument("--lr", default=1e-4, type=float, help="Learning rate")
optimizer_group.add_argument("--wd", default=2e-5, type=float, help="Weight decay")
optimizer_group.add_argument("--clip", default=0.5, type=float, help="Clip norm")
optimizer_group.add_argument("--momentum", default=0.9, type=float, help="Momentum")
optimizer_group.add_argument("--beta1", default=0.9, type=float, help="Beta1 for optimizer")
optimizer_group.add_argument("--beta2", default=0.99, type=float, help="Beta2 for optimizer")

# Training parameters
train_group = parser.add_argument_group("Training parameters")
train_group.add_argument("--multi-scale", default=[1], nargs="+", type=float, help="Multi-scale training")

# Utils parameters
utils_group = parser.add_argument_group("Utils parameters")
utils_group.add_argument('--seed', type=int, default=42, help='Random seed for repudctivity')
utils_group.add_argument('--log-interval', type=int, default=50, help='Frequency of logging output')
utils_group.add_argument('--output-dir', type=str, default="res/", help='path to save checkpoint')
utils_group.add_argument('--monitor', type=str, default="_dice", help='metrics for monitoring model performance')
utils_group.add_argument('--monitor-direction', type=str, default="max", help='max or min')

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
transform_group.add_argument("--img-size", type=int, default=352, help="Image size (default: 352)")
transform_group.add_argument("--crop", type=int, default=0, help="Crop parameter (default: 0)")
transform_group.add_argument("--hflip", type=float, default=0.5, help="Horizontal flip parameter (default: 0.5)")
transform_group.add_argument("--vflip", type=float, default=0.5, help="Vertical flip parameter (default: 0.5)")
transform_group.add_argument("--jitter", type=float, default=0.5, help="Jitter parameter (default: 0.5)")
transform_group.add_argument("--unsharp", type=float, default=0.5, help="Unsharp parameter (default: 0.5)")
transform_group.add_argument("--cutout", type=float, default=0, help="Cutout parameter (default: 0)")
transform_group.add_argument("--distortion", type=float, default=0.5, help="Distortion parameter (default: 0.5)")
transform_group.add_argument("--rotate", type=float, default=0.5, help="Rotate parameter (default: 0.5)")
transform_group.add_argument("--scale", type=float, default=0.5, help="Scaling parameter (default: 0.5)")
transform_group.add_argument("--affine", type=float, default=0.5, help="Affine parameter (default: 0.5)")

# Wandb Logger parameters
wandb_group = parser.add_argument_group("W & B parameters")
wandb_group.add_argument('--project', type=str, default="Polyp-Segmentation")
wandb_group.add_argument('--entity', type=str, default="Your Entity")
wandb_group.add_argument('--name', type=str, default="Polyp-model")
wandb_group.add_argument('--wandb', default=False, action="store_true")
wandb_group.add_argument('--logger_train', default=False, action="store_true")
wandb_group.add_argument('--log_image', default=False, action="store_true")

parser.add_argument("--rank", type=int, default=0, help="training device")

parser.add_argument("-b", "--batch-size", default=32, type=int)
parser.add_argument("--amp", default=False, action="store_true",
                    help="Use torch.cuda.amp for mixed precision training")
args, unknown = parser.parse_known_args()

os.makedirs(args.output_dir, exist_ok=True)

def main(args):

    utils.print_with_timestamp(True)

    device = f"cuda:{args.rank}"
    utils.seed_everything(args.seed)
    num_classes = args.num_classes + 1

    model = create_model(args.model)
    model = torch.nn.DataParallel(model, device_ids=[0,1,2,3])
    model.to(device)

    args.is_master = (args.rank % 8) == 0

    wandb_logger = None
    if args.wandb and args.is_master:
        wandb_logger = utils.WandbLogger(args)
        wandb_logger.login()
        wandb_logger.init_run()

    args.max_lr = args.max_lr if args.max_lr else args.lr
    args.img_mode = "RGB" if args.in_chans == 3 else "L"

    reader_train = ImageFolderReader(os.path.join(args.data_dir, args.train_split))
    reader_valid = ImageFolderReader(os.path.join(args.data_dir, args.val_split))
    reader_test = ImageFolderReader(os.path.join(args.data_dir, args.test_split))

    dataset_train = create_dataset(
        name=args.dataset,
        root=args.data_dir,
        split=args.train_split,
        reader=reader_train,
        img_mode=args.img_mode,
    )

    dataset_valid = create_dataset(
        name=args.dataset,
        root=args.data_dir,
        split=args.val_split,
        reader=reader_valid,
        img_mode=args.img_mode,
    )

    dataset_test = create_dataset(
        name=args.dataset,
        root=args.data_dir,
        split=args.test_split,
        reader=reader_test,
        img_mode=args.img_mode,
    )

    loader_train = create_loader(
        dataset_train,
        args,
        is_training=True,
    )

    loader_valid = create_loader(
        dataset_valid,
        args,
        is_training=False,
    )

    loader_test = create_loader(
        dataset_test,
        args,
        is_training=False,
    )

    if wandb_logger and args.log_image:
        iterator_train = utils.log_batch_data(loader_train)
        for idx, data in enumerate(iterator_train):
            wandb_logger.log_image(f"{args.train_split}-batch-{idx}", *data)

        iterator_valid = utils.log_batch_data(loader_valid)
        for idx, data in enumerate(iterator_valid):
            wandb_logger.log_image(f"{args.val_split}-batch-{idx}", *data)

        iterator_test = utils.log_batch_data(loader_test)
        for idx, data in enumerate(iterator_test):
            wandb_logger.log_image(f"{args.test_split}-batch-{idx}", *data)
        wandb_logger._wandb.log({"Image table": wandb_logger.table})

    params_to_optimize = [
        {"params": [p for p in model.parameters() if p.requires_grad]},
    ]

    optimizer = create_optimizer(params_to_optimize, args)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    steps_per_epoch = len(loader_train)

    lr_scheduler = create_scheduler(
        optimizer=optimizer, 
        args=args, 
        steps_per_epoch=steps_per_epoch, 
        step_on_epochs=False
    )

    criterion = create_criterion(args=args)

    n_parameters = sum(p.numel() for p in params_to_optimize[0]["params"] if p.requires_grad)
    
    print("Training augmentation:", dataset_train.transform)
    print("Validation augmentation:", dataset_valid.transform)
    print('number of params (M): %.2f' % (n_parameters / 1.e6))
    print("LR = %.8f" % args.lr)
    print("WD = %.8f" % args.wd)
    print("Grad Clip Norm = %.8f" % args.clip)
    print("Batch size = %d" % args.batch_size)
    print("Number of training epochs = %d" % args.epochs)
    print("Number of training examples = %d" % len(dataset_train))
    print("Number of training steps per epoch = %d" % steps_per_epoch)
    print("Use %s scheduler" % str(args.scheduler_name))
    print("Use %s optimizer" % str(optimizer.__class__.__name__))
    print("Use %s criterion" % str(args.criterion))

    best_metrics = 0
    for epoch in range(args.epochs):

        logger_train, confmat_train = train_one_epoch(
            model, optimizer, loader_train, device, epoch, criterion,
            lr_scheduler=lr_scheduler, clip=args.clip, multi_scale=args.multi_scale, 
            num_classes=num_classes, print_freq=args.log_interval, scaler=scaler
        )
        
        logger_valid, confmat_valid = evaluate(
            model, loader_valid, device=device, 
            num_classes=num_classes, criterion=criterion
        )

        if wandb_logger:
            if args.logger_train:
                wandb_logger._wandb.log({f"LR": getattr(logger_train, "lr", -1).value, "epoch": epoch})
                wandb_logger._wandb.log({f"[Train] Loss": getattr(logger_train, "loss", -1).value, "epoch": epoch})
                wandb_logger._wandb.log({f"[Train] Precision": getattr(confmat_train, "mprecision", -1), "epoch": epoch})
                wandb_logger._wandb.log({f"[Train] Dice": getattr(confmat_train, "mdice", -1), "epoch": epoch})
                wandb_logger._wandb.log({f"[Train] mIoU": getattr(confmat_train, "miou", -1), "epoch": epoch})
                wandb_logger._wandb.log({f"[Train] Recall": getattr(confmat_train, "mrecall", -1), "epoch": epoch})

            wandb_logger._wandb.log({f"Validation Loss": getattr(logger_valid, "loss", -1).value, "epoch": epoch})
            wandb_logger._wandb.log({f"[Val] Precision": getattr(confmat_valid, "mprecision", -1), "epoch": epoch})
            wandb_logger._wandb.log({f"[Val] Dice": getattr(confmat_valid, "mdice", -1), "epoch": epoch})
            wandb_logger._wandb.log({f"[Val] mIoU": getattr(confmat_valid, "miou", -1), "epoch": epoch})
            wandb_logger._wandb.log({f"[Val] Recall": getattr(confmat_valid, "mrecall", -1), "epoch": epoch})
    
        save_file = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
        }
        if args.amp:
            save_file["scaler"] = scaler.state_dict()

        metrics = getattr(confmat_valid, args.monitor, None)
        if args.monitor_direction == "max":
            if metrics > best_metrics:
                best_metrics = metrics
                print("Saving ...")
                torch.save(save_file, os.path.join(args.output_dir, "best_ckpt.pth"))
        else:
            if metrics < best_metrics:
                best_metrics = metrics
                print("Saving ...")
                torch.save(save_file, os.path.join(args.output_dir, "best_ckpt.pth"))

        torch.save(save_file, os.path.join(args.output_dir, "last_ckpt.pth"))

    if wandb_logger:
        wandb_logger._wandb.save(os.path.join(args.output_dir, "best_ckpt.pth"))
        wandb_logger._wandb.save(os.path.join(args.output_dir, "last_ckpt.pth"))


    ##### Validation
    best_ckpt = torch.load(os.path.join(args.output_dir, "best_ckpt.pth"), map_location=device)
    model.load_state_dict(best_ckpt["model"])
    logger_valid, confmat_valid = evaluate(
        model, loader_valid, device=device, 
        num_classes=num_classes, criterion=criterion
    )        
    if wandb_logger:
        wandb_logger._wandb.log({f"Best [Val] Loss": getattr(logger_valid, "loss", -1).value})
        wandb_logger._wandb.log({f"Best [Val] Precision": getattr(confmat_valid, "mprecision", -1)})
        wandb_logger._wandb.log({f"Best [Val] Dice": getattr(confmat_valid, "mdice", -1)})
        wandb_logger._wandb.log({f"Best [Val] mIoU": getattr(confmat_valid, "miou", -1)})
        wandb_logger._wandb.log({f"Best [Val] Recall": getattr(confmat_valid, "mrecall", -1)})
    
    ##### Testing
    if args.test_split:
        logger_test, confmat_test = evaluate(
            model, loader_test, device=device, 
            num_classes=num_classes, criterion=criterion
        )        
        if wandb_logger:
            wandb_logger._wandb.log({f"Test Loss": getattr(logger_test, "loss", -1).value})
            wandb_logger._wandb.log({f"[Test] Acc": getattr(confmat_test, "mprecision", -1)})
            wandb_logger._wandb.log({f"[Test] Dice": getattr(confmat_test, "mdice", -1)})
            wandb_logger._wandb.log({f"[Test] mIoU": getattr(confmat_test, "miou", -1)})
            wandb_logger._wandb.log({f"[Test] Recall": getattr(confmat_test, "mrecall", -1)})

if __name__ == '__main__':
    main(args)