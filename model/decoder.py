import torch 
import torch.nn as nn
import segmentation_models_pytorch as smp 
from timm.models.registry import register_model

class Unet(nn.Module):
    def __init__(self, decoder, size=352, **kwargs):
        super().__init__()

        self.decoder = decoder

        if self.decoder == "unet":
            self.backbone = smp.Unet(
                encoder_name="resnet50",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
                in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=1,                      # model output channels (number of classes in your dataset)
            )
        elif self.decoder == "unetplusplus":
            self.backbone = smp.UnetPlusPlus(
                encoder_name="resnet50",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
                in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=1,                      # model output channels (number of classes in your dataset)
            )
        elif self.decoder == "manet":
            self.backbone = smp.MAnet(
                encoder_name="resnet50",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
                in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=1,                      # model output channels (number of classes in your dataset)
            )
        elif self.decoder == "linknet":
            self.backbone = smp.Linknet(
                encoder_name="resnet50",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
                in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=1,                      # model output channels (number of classes in your dataset)
            )
        elif self.decoder == "fpn":
            self.backbone = smp.FPN(
                encoder_name="resnet50",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
                in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=1,                      # model output channels (number of classes in your dataset)
            )
        elif self.decoder == "pspnet":
            self.backbone = smp.PSPNet(
                encoder_name="resnet50",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
                in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=1,                      # model output channels (number of classes in your dataset)
            )
        elif self.decoder == "pan":
            self.backbone = smp.PAN(
                encoder_name="resnet50",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
                in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=1,                      # model output channels (number of classes in your dataset)
            )
        elif self.decoder == "deeplabv3":
            self.backbone = smp.DeepLabV3(
                encoder_name="resnet50",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
                in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=1,                      # model output channels (number of classes in your dataset)
            )
        elif self.decoder == "deeplabv3plus":
            self.backbone = smp.DeepLabV3Plus(
                encoder_name="resnet50",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
                in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=1,                      # model output channels (number of classes in your dataset)
            )

    def forward(self, x):
        x = self.backbone(x)

        return x
    
@register_model
def unet(pretrained=False, **kwargs):
    model = Unet(decoder="unet", **kwargs)
    return model
    
@register_model
def unetplusplus(pretrained=False, **kwargs):
    model = Unet(decoder="unetplusplus", **kwargs)
    return model
    
@register_model
def manet(pretrained=False, **kwargs):
    model = Unet(decoder="manet", **kwargs)
    return model
    
@register_model
def linknet(pretrained=False, **kwargs):
    model = Unet(decoder="linknet", **kwargs)
    return model
    
@register_model
def fpn(pretrained=False, **kwargs):
    model = Unet(decoder="fpn", **kwargs)
    return model
    
@register_model
def pspnet(pretrained=False, **kwargs):
    model = Unet(decoder="pspnet", **kwargs)
    return model
    
@register_model
def pan(pretrained=False, **kwargs):
    model = Unet(decoder="pan", **kwargs)
    return model
    
@register_model
def deeplabv3(pretrained=False, **kwargs):
    model = Unet(decoder="deeplabv3", **kwargs)
    return model
    
@register_model
def deeplabv3plus(pretrained=False, **kwargs):
    model = Unet(decoder="deeplabv3plus", **kwargs)
    return model