import torch
from PIL import Image
import numpy as np
from typing import Callable, Optional

class ImageDataset(torch.utils.data.Dataset):
    """
    A custom PyTorch dataset for handling image data.

    Parameters:
    - root (str): The root path of the dataset.
    - split (str): The split of the dataset (e.g., 'train', 'val', 'test').
    - reader (list): A list of dictionaries containing information about each image.
    - class_map (dict): A dictionary mapping class names to indices.
    - img_mode (str): The mode to use for loading images (e.g., 'RGB', 'L').
    - transform (Optional[Callable]): An optional transform to be applied to the image.
    - target_transform (Optional[Callable]): An optional transform to be applied to the target class index.
    """

    def __init__(self, root: str, split: str, reader: list, img_mode: str,
                 transform: Optional[Callable] = None):
        self.root = root
        self.split = split
        self.reader = reader
        self.img_mode = img_mode
        self.transform = transform

    def __getitem__(self, index):
        """
        Get the image and its corresponding class index at the specified index.

        Parameters:
        - index (int): The index of the sample to retrieve.

        Returns:
        - img (PIL.Image.Image): The loaded image.
        - class_idx (int): The class index of the image.
        """
        img_data = self.reader[index]
        img_path = img_data["image_path"]
        mask_path = img_data["mask_path"]

        img = np.array(Image.open(img_path).convert(self.img_mode))
        mask = np.array(Image.open(mask_path).convert("L"))

        mask = (mask > 127) * 1.

        if self.transform is not None:
            transformed = self.transform(image=img, mask=mask)
            img, mask = transformed['image'], transformed['mask']

        return img, mask[None]

    def __len__(self):
        """
        Get the number of samples in the dataset.

        Returns:
        - int: The number of samples in the dataset.
        """
        return len(self.reader)