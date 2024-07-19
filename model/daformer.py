import torch
import torch.nn as nn
import torch.nn.functional as F

class MixUpSample(nn.Module):
	def __init__( self, scale_factor=2):
		super().__init__()
		assert(scale_factor!=1)
		
		self.mixing = nn.Parameter(torch.tensor(0.5))
		self.scale_factor = scale_factor
	
	def forward(self, x):
		x = self.mixing *F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False) \
			+ (1-self.mixing )*F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')
		return x

#https://github.com/lhoyer/DAFormer/blob/master/mmseg/models/decode_heads/daformer_head.py
def Conv2dBnReLU(in_channel, out_channel, kernel_size=3, padding=1,stride=1, dilation=1):
	return nn.Sequential(
		nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, bias=False),
		nn.BatchNorm2d(out_channel),
		nn.ReLU(inplace=True),
	)

class DaformerDecoder(nn.Module):
	def __init__(
			self,
			encoder_dim = [32, 64, 160, 256],
			decoder_dim = 256,
			dilation = [1, 6, 12, 18],
			use_bn_mlp  = True,
			fuse = 'conv3x3',
	):
		super().__init__()
		self.mlp = nn.ModuleList([
			nn.Sequential(
				# Conv2dBnReLU(dim, decoder_dim, 1, padding=0), #follow mmseg to use conv-bn-relu
				*(
				  ( nn.Conv2d(dim, decoder_dim, 1, padding= 0,  bias=False),
					nn.BatchNorm2d(decoder_dim),
					nn.ReLU(inplace=True),
				)if use_bn_mlp else
				  ( nn.Conv2d(dim, decoder_dim, 1, padding= 0,  bias=True),)
				),
				
				MixUpSample(2**i) if i!=0 else nn.Identity(),
			) for i, dim in enumerate(encoder_dim)])
	  
		if fuse=='conv1x1':
			self.fuse = nn.Sequential(
				nn.Conv2d(len(encoder_dim) * decoder_dim, decoder_dim, 1, padding=0, bias=False),
				nn.BatchNorm2d(decoder_dim),
				nn.ReLU(inplace=True),
			)
		
		if fuse=='conv3x3':
			self.fuse = nn.Sequential(
				nn.Conv2d(len(encoder_dim) * decoder_dim, decoder_dim, 3, padding=1, bias=False),
				nn.BatchNorm2d(decoder_dim),
				nn.ReLU(inplace=True),
			)
		
	def forward(self, feature):

		out = []
		for i,f in enumerate(feature):
			f = self.mlp[i](f)
			out.append(f)
		x = self.fuse(torch.cat(out, dim = 1))
		return x, out


class daformer_conv3x3 (DaformerDecoder):
	def __init__(self, **kwargs):
		super(daformer_conv3x3, self).__init__(
			fuse = 'conv3x3',
			**kwargs
		)
class daformer_conv1x1 (DaformerDecoder):
	def __init__(self, **kwargs):
		super(daformer_conv1x1, self).__init__(
			fuse = 'conv1x1',
			**kwargs
		)
 