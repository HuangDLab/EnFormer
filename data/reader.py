import os
from typing import List, Dict, Tuple

def read_images_from_folder(root: str, **kwargs) -> List[Dict]:
    """
    Read images from subfolders within the specified root folder and associate each image with its class.

    Parameters:
    - root (str): The path to the root folder containing subfolders for each class.

    Returns:
    - image_data (list): A list of dictionaries, each containing information about an image (path, class, class index, and image object).
    """

    image_data = []
    image_paths = os.path.join(root, "images")
    mask_paths = os.path.join(root, "masks")

    for path in os.listdir(image_paths):
        if path.lower().endswith(('.png', '.jpg', '.bmp')):
            image_id = path[:-4]
            name = path.split("/")[-1]
            image_path = os.path.join(image_paths, path)
            mask_path = os.path.join(mask_paths, path)

            image_data.append({
                'image_path': image_path,
                'mask_path': mask_path,
                "image_id": image_id,
                "name": name
            })

    return image_data

class ImageFolderReader:

    def __init__(self, root: str, **kwargs):

        self.root = root
        self.image_data = read_images_from_folder(root=root)
        
        self.__sanity_check()

    def __getitem__(self, index: int) -> dict:
        return self.image_data[index]
    
    def __len__(self) -> int:
        return len(self.image_data)
    
    def __sanity_check(self) -> None:

        if len(self.image_data) == 0:
            raise RuntimeError(
                f"Found 0 images in subfolders of {self.root}."
            )
        
        split = self.root.split("/")[-1]

        print(f"[{split}] Found {len(self.image_data)} samples")

