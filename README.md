# EnFormer

This project is developed based on the timm framework [https://github.com/huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models) specifically for Polyp Segmentation.

## Backbone weight

Pretrained weight for transformer backbones can be downloaded via:

```bash
wget https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b3.pth
wget https://vcl.ucsd.edu/coat/pretrained/coat_lite_mini_6b4a8ae5.pth
wget https://vcl.ucsd.edu/coat/pretrained/coat_lite_small_8d362f48.pth
wget https://vcl.ucsd.edu/coat/pretrained/coat_lite_medium_384x384_f9129688.pth
```

## Data

Your data format should be as follows:

```
-------------------
[your dataset root]
-- train /
---- images /
------ image1.jpg
---- masks /
------ mask1.jpg
-- val /
---- images /
------ image1.jpg
---- masks /
------ mask1.jpg
-- test /
---- images /
------ image1.jpg
---- masks /
------ mask1.jpg
-------------------
```

- [your dataset root] is the root directory of your data, which needs to be modified in data-dir in args
- Data is divided into three folders: train/val/test. The names do not have to be unique, but you must specify them in train-split/val-split/test-split in args according to your settings
- `images` is used to store all images, and `masks` is used to store all segmentation labels. The folder names are fixed and cannot be changed arbitrarily. Also, the image names in the `images` folder (e.g., `images/something.jpg`) should correspond to the segmentation labels in the `masks` folder (e.g., `masks/something.jpg`)

Training and testing dataset can be downloaded from:

- TrainDataset: https://drive.google.com/file/d/1YiGHLw4iTvKdvbT6MgwO9zcCv8zJ_Bnb/view
- TestDataset: https://drive.google.com/file/d/1Y2z7FD5p5y31vkZwQQomXFRB0HutHyao/view

This dataset already includes a testing set, so we split the TrainDataset into train/val, and copy Kvasir from the TestDataset into TrainDataset, renaming it to test.

Your directory structure should be:

```
-------------------
TrainDataset /
-- train
-- val
-- test
TestDataset /
-- CVC-300
-- CVC-ClinicDB
-- CVC-ColonDB
-- ETIS-LaribPolypDB
-- Kvasir
-------------------
```

And the settings in args should be:

```
--data-dir [path to TrainDataset]/
--train-split train
--val-split val
--test-split test
```

## Usage

```bash
pip install -r requirements.txt
bash train.sh
```

## Metrics calculation

Please refer to [https://github.com/DengPingFan/PraNet](https://github.com/DengPingFan/PraNet)

## Pre-computed maps

[[Google drive]](https://drive.google.com/file/d/1kA3cydMnJB8-zucG1pP6RrF1wU9tttxO/view?usp=sharing)

## Pre-trained weight

Pre-trained weight can be downloaded via: [[Google drive]](https://drive.google.com/file/d/1UsRR8ggtRWUwWVLBrGF_fAvciM0VCqn2/view?usp=sharing)
