import torch
import torch.nn.functional as F
from .metrics import SmoothedValue, MetricLogger, ConfusionMatrix

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def train_one_epoch(model, optimizer, data_loader, device, epoch, criterion, lr_scheduler, clip, multi_scale, num_classes, print_freq=10, scaler=None):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    multi_scale = multi_scale if multi_scale is not None else [1]

    confmat = ConfusionMatrix(num_classes)
    for image, mask in metric_logger.log_every(data_loader, print_freq, header):
        for scale in multi_scale:
            image, mask = image.to(device), mask.to(device).float()

            if scale != 1:
                img_size = image.size(-1)
                resize = int(round(img_size * scale / 32) * 32)
                image = F.interpolate(image, size=(resize, resize), mode='bilinear', align_corners=True)
                mask = F.interpolate(mask, size=(resize, resize), mode='bilinear', align_corners=True)

            with torch.cuda.amp.autocast(enabled=scaler is not None):
                output = model(image)
                loss = criterion(output, mask)

            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                if clip:
                    clip_gradient(optimizer, clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if clip:
                    clip_gradient(optimizer, clip)
                optimizer.step()

            if scale == 1:
                confmat.update(mask.flatten(), ((output>0)*1).flatten())
                confmat.step()
                confmat.reset()
                metric_logger.update(loss=loss.item())
                metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        lr_scheduler.step()

    return metric_logger, confmat