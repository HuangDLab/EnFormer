from .dataset import ImageDataset

def create_dataset(
        name,
        root,
        split,
        reader=None,
        img_mode="RGB",
        **kwargs
):
    name = name.lower()
    if True:
        ds = ImageDataset(
            root=root,
            split=split,
            reader=reader,
            img_mode=img_mode,
            **kwargs
        )
    return ds