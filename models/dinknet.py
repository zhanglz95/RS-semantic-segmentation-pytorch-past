import torch
import torch.nn as nn
from torchvision import models

class Dblock(nn.Module):
	def __init__(self, channel):
		super(Dblock, self).__init__()
		self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
		self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
		self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
		self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)

		self.relu = nn.ReLU(inplace=True)

		for m in self.modules():
			if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
				if m.bias is not None:
					m.bias.data.zero_()

	def forward(self, x):
		dilate1_out = self.relu(self.dilate1(x))
		dilate2_out = self.relu(self.dilate2(dilate1_out))
		dilate3_out = self.relu(self.dilate3(dilate2_out))
		dilate4_out = self.relu(self.dilate4(dilate3_out))

		out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
		return out

class DecoderBlock(nn.Module):
	def __init__(self, in_channels, n_filters):
		super(DecoderBlock, self).__init__()
		self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
		self.norm1 = nn.BatchNorm2d(in_channels // 4)
		self.relu1 = nn.ReLU(inplace=True)

		self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
		self.norm2 = nn.BatchNorm2d(in_channels // 4)
		self.relu2 = nn.ReLU(inplace=True)

		self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
		self.norm3 = nn.BatchNorm2d(n_filters)
		self.relu3 = nn.ReLU(inplace=True)

	def forward(self, x):
		x = self.conv1(x)
		x = self.norm1(x)
		x = self.relu1(x)
		x = self.deconv2(x)
		x = self.norm2(x)
		x = self.relu2(x)
		x = self.conv3(x)
		x = self.norm3(x)
		x = self.relu3(x)

		return x


class DinkNet34(nn.Module):
	def __init__(self, num_classes = 1):
		super(DinkNet34, self).__init__()

		resnet = models.resnet34(pretrained=True)

		self.firstconv = resnet.conv1
		self.firstbn = resnet.bn1
		self.firstrelu = resnet.relu
		self.firstmaxpool = resnet.maxpool
		self.down1 = resnet.layer1
		self.down2 = resnet.layer2
		self.down3 = resnet.layer3
		self.down4 = resnet.layer4

		self.dblock = Dblock(512)

		self.up4 = DecoderBlock(512, 256)
		self.up3 = DecoderBlock(256, 128)
		self.up2 = DecoderBlock(128, 64)
		self.up1 = DecoderBlock(64, 64)

		self.finaldeconv1 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
		self.finalrelu1 = nn.ReLU(inplace=True)
		self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
		self.finalrelu2 = nn.ReLU(inplace=True)
		self.finalconv3 = nn.Conv2d(32, num_classes + 1, 3, padding=1)

	def forward(self, x):
		x = self.firstconv(x)
		x = self.firstbn(x)
		x = self.firstrelu(x)
		x = self.firstmaxpool(x)

		down1_out = self.down1(x)
		down2_out = self.down2(down1_out)
		down3_out = self.down3(down2_out)
		down4_out = self.down4(down3_out)

		center = self.dblock(down4_out)

		up4_out = self.up4(center) + down3_out
		up3_out = self.up3(up4_out) + down2_out
		up2_out = self.up2(up3_out) + down1_out
		up1_out = self.up1(up2_out)

		out = self.finaldeconv1(up1_out)
		out = self.finalrelu1(out)
		out = self.finalconv2(out)
		out = self.finalrelu2(out)
		out = self.finalconv3(out)

		return out		