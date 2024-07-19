import torch
from .metrics import SmoothedValue, MetricLogger, ConfusionMatrix

@torch.no_grad()
def evaluate(model, data_loader, device, num_classes, criterion):
    model.eval()
    confmat = ConfusionMatrix(num_classes)
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'

    for image, mask in metric_logger.log_every(data_loader, 100, header):
        image, mask = image.to(device), mask.to(device).float()
        output = model(image)
        loss = criterion(output, mask)


        metric_logger.update(loss=loss.item())
        
        confmat.update(mask.flatten(), ((output>0)*1).flatten())
        confmat.step()
        confmat.reset()
    
    confmat.summary()
    confmat.reduce_from_all_processes()

    return metric_logger, confmat