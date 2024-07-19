import wandb
import torch 
import numpy as np 
import torchvision
from PIL import Image

class WandbLogger(object):
    
    def __init__(self, config):
        
        self.config = config
        self._wandb =  wandb
        
    def login(self):
        self._wandb.login(key="your api key", relogin=True)
        
    def init_run(self):
        self.table = wandb.Table(columns=["Categorical", "Images", "Masks", "Fuses"], allow_mixed_types=True)
        self._wandb.init(
            config=self.config,
            project=self.config.project,
            entity=self.config.entity,
            name=self.config.name,
            allow_val_change=True,
            reinit=True
        )
        
    def log_image(self, title, images, masks, fuses):
        self.table.add_data(
            title,
            wandb.Image(images),
            wandb.Image(masks),
            wandb.Image(fuses),
        )


def log_batch_data(loader):
    for idx, (imgs, masks) in enumerate(loader):
        bs = imgs.size(0)

        img_grid = torchvision.utils.make_grid(imgs, nrow=int(bs // 4), normalize=True)
        mask_grid = torchvision.utils.make_grid(torch.cat([masks, masks, masks], dim=1), nrow=int(bs // 4), normalize=False)

        img_grid = (img_grid.numpy().transpose(1,2,0) * 255).astype(np.uint8)
        mask_grid = (mask_grid.numpy().transpose(1,2,0) * 255).astype(np.uint8)

        img_grid = Image.fromarray(img_grid)
        mask_grid = Image.fromarray(mask_grid)

        fusion_grid = Image.blend(img_grid, mask_grid, alpha=0.5)

        if idx < 3:
            yield img_grid, mask_grid, fusion_grid
        else:
            break  # Stop yielding after the first 5 batches