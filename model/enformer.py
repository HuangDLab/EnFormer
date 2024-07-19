import torch 
import torch.nn as nn
import timm
from timm.models.registry import register_model
from .daformer import daformer_conv3x3
from .fcbformer import RB, TB, FCB, CB
from .coat import coat_lite_mini

class FeatureInfo:
    def __init__(self, model):
        self.model = model

    def channels(self):
        if self.model == "coat_lite_mini":
            return [64, 128, 320, 512]
        elif self.model == "coat_lite_small":
            return [64, 128, 320, 512]
        elif self.model == "coat_lite_medium":
            return [128, 256, 320, 512]
        else:
            raise ValueError("Invalid model identifier")
    
class CoaTB(nn.Module):

    def __init__(self, model_name, feature_info, out_channels=64, pretrained_path=None):
        super().__init__()

        self.model = timm.create_model(
            model_name,
            return_interm_layers=True, 
            out_features=["x1_nocls", "x2_nocls", "x3_nocls", "x4_nocls"]
        )
        self.model.feature_info = feature_info(model_name)

        if pretrained_path:
            state_dict = torch.load(pretrained_path, map_location="cpu")
            self.model.load_state_dict(state_dict["model"], strict=False)

        self.LE = nn.ModuleList([])
        for i in range(len(self.model.feature_info.channels())):
            self.LE.append(
                nn.Sequential(
                    RB(self.model.feature_info.channels()[i], out_channels), RB(out_channels, out_channels)
                )
            )

    def forward(self, x):
        x = self.model(x)
        out = []
        for (k, v), layer in zip(x.items(), self.LE):
            out.append(layer(v))
        return out
    
class TimmModel(nn.Module):

    def __init__(self, model_name, out_channels=64, **kwargs):
        super().__init__()

        self.model = timm.create_model(model_name, num_classes=0, features_only=True, pretrained=True)

        self.LE = nn.ModuleList([])
        for i in range(4):
            self.LE.append(
                nn.Sequential(
                    RB(self.model.feature_info.channels()[-4:][i], out_channels), RB(out_channels, out_channels)
                )
            )

    def forward(self, x):
        x = self.model(x)
        x = x[-4:]
        out = []
        for v, layer in zip(x, self.LE):
            out.append(layer(v))
        return out
    
class CoaTBMini(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = coat_lite_mini(return_interm_layers=True, out_features=["x1_nocls", "x2_nocls", "x3_nocls", "x4_nocls"])

        self.LE = nn.ModuleList([])
        for i in range(4):
            self.LE.append(
                nn.Sequential(
                    RB([64, 128, 320, 512][i], 64), RB(64, 64)
                )
            )

    def forward(self, x):
        x = self.model(x)
        out = []
        for (k, v), layer in zip(x.items(), self.LE):
            print(v.shape)
            out.append(layer(v))
        return out
    
class EnFormerLite(nn.Module):
    def __init__(self, transformer_name, convolution_name, size=352, out_channels=64, pretrained_path=None, **kwargs):
        super().__init__()

        self.TB = CoaTB(transformer_name, FeatureInfo, out_channels, pretrained_path)
        # self.TB = CoaTBMini()
        if convolution_name == "cb":
            self.FCB = CB()
        elif convolution_name in timm.list_models():
            self.FCB = TimmModel(convolution_name, out_channels, **kwargs)
        else:
            raise ValueError(f"Invalid convolution name: {convolution_name}")
        self.PH = nn.Sequential(
            RB(out_channels, out_channels), RB(out_channels, out_channels), nn.Conv2d(out_channels, 1, kernel_size=1)
        )
        self.up_tosize = nn.Upsample(size=size)
        self.decoder = daformer_conv3x3(
            encoder_dim = [out_channels * 2 for _ in range(4)],
            decoder_dim = out_channels,
        )

    def forward(self, x):
        x1_feats = self.TB(x)
        x2_feats = self.FCB(x)
        x, _ = self.decoder(list(map(lambda a, b: torch.cat([a, b], dim=1), x1_feats, x2_feats)))
        x = self.up_tosize(x)
        out = self.PH(x)

        return out
    
class EnFormer(nn.Module):
    def __init__(self, size=352, **kwargs):

        super().__init__()

        self.TB = TB(out_feat=True)

        self.FCB = FCB(in_resolution=size, min_channel_mults=[1,1,2,2,4,4], out_feat=True)
        self.PH = nn.Sequential(
            RB(64 + 32 + 64, 64), RB(64, 64), nn.Conv2d(64, 1, kernel_size=1)
        )
        self.up_tosize = nn.Upsample(size=size)
        self.decoder = daformer_conv3x3(
            encoder_dim = [128,128,128,128],
            decoder_dim = 64,
            dilation = None,
        )

        self.LE2 = nn.ModuleList([])
        for i in range(4):
            self.LE2.append(
                nn.Sequential(
                    RB([64,64,128,128][i], 64), RB(64, 64)
                )
            )
        
        self.LE1 = nn.ModuleList([])
        for i in range(4):
            self.LE1.append(
                nn.Sequential(
                    RB([64,128,320,512][i], 64), RB(64, 64)
                )
            )

    def forward(self, x):
        x1, x1_feats = self.TB(x)
        x2, x2_feats = self.FCB(x)
        x1_feats = list(map(lambda x, i: self.LE1[i](x), x1_feats, range(4)))
        x2_feats = list(map(lambda x, i: self.LE2[i](x), x2_feats, range(4)))
        x3, _ = self.decoder(list(map(lambda a, b: torch.cat([a, b], dim=1), x1_feats, x2_feats)))
        x1 = self.up_tosize(x1)
        x3 = self.up_tosize(x3)
        x = torch.cat((x1, x2, x3), dim=1)
        out = self.PH(x)

        return out
    
@register_model
def enformer_lite_mini(pretrained_path=False, **kwargs):
    model = EnFormerLite(transformer_name="coat_lite_mini", convolution_name="cb", size=352, out_channels=64, pretrained_path=pretrained_path, **kwargs)
    return model

@register_model
def enformer_lite_small(pretrained_path=False, **kwargs):
    model = EnFormerLite(transformer_name="coat_lite_small", convolution_name="cb", size=352, out_channels=64, pretrained_path=pretrained_path, **kwargs)
    return model

@register_model
def enformer_lite_medium(pretrained_path=False, **kwargs):
    model = EnFormerLite(transformer_name="coat_lite_medium", convolution_name="cb", size=352, out_channels=64, pretrained_path=pretrained_path, **kwargs)
    return model

@register_model
def enformer_lite_large(pretrained_path=False, **kwargs):
    model = EnFormerLite(transformer_name="coat_lite_medium", convolution_name="resnet50", size=352, out_channels=64, pretrained_path=pretrained_path, **kwargs)
    return model
    
@register_model
def enformer(pretrained=False, **kwargs):
    model = EnFormer(**kwargs)
    return model