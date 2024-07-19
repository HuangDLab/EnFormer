import torch

def poly_lr_scheduler(optimizer, num_step, epochs):
    def poly(x):
        return (1 - (x) / (num_step * epochs)) ** 0.9
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=poly)