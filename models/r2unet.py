import torch
import torch.nn as nn
import torch.nn.functional as F

class conv_block(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(conv_block,self).__init__()
        self.one_stage=nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False)
        )

    def forward(self,x):
        return self.one_stage(x)

class up_conv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(up_conv,self).__init__()
        self.up=nn.Sequential(
            nn.Upsample(scale_factor=2),
            conv_block(in_channels=in_channels,out_channels=out_channels)
        )
    def forward(self,x):
        return self.up(x)


class recurrent_block(nn.Module):
    def __init__(self,channel,t=2):
        super(recurrent_block,self).__init__()
        self.t=t
        self.conv=conv_block(channel,channel)
        self.conv_1x1=nn.Conv2d(in_channels=channel,out_channels=channel,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        x=self.conv_1x1(x)
        for i in range(self.t):
            x1=self.conv(x) if i==0 else self.conv(x1+x)
        return x1

class residual_block(nn.Module):
    def __init__(self,in_channels,out_channels,t=2):
        super(residual_block,self).__init__()
        self.rcnn=nn.Sequential(
            recurrent_block(channel=out_channels,t=t),
            recurrent_block(channel=out_channels, t=t)
        )
        self.conv_1x1=nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        x=self.conv_1x1(x)
        x1=self.rcnn(x)
        x1+=x
        return x1


class R2_Unet(nn.Module):
    def __init__(self,num_classes=1):
        super(R2_Unet, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.residual_rcnn1 = residual_block(in_channels=3,out_channels=64,t=2)
        self.residual_rcnn2 = residual_block(in_channels=64, out_channels=128, t=2)
        self.residual_rcnn3 = residual_block(in_channels=128, out_channels=256, t=2)
        self.residual_rcnn4 = residual_block(in_channels=256, out_channels=512, t=2)
        self.residual_rcnn5 = residual_block(in_channels=512, out_channels=1024, t=2)

        self.up5 = up_conv(in_channels=1024,out_channels=512)
        self.up4 = up_conv(in_channels=512,out_channels=256)
        self.up3 = up_conv(in_channels=256,out_channels=128)
        self.up2 = up_conv(in_channels=128,out_channels=64)

        self.up_resdual5 = residual_block(in_channels=1024,out_channels=512,t=2)
        self.up_resdual4 = residual_block(in_channels=512, out_channels=256, t=2)
        self.up_resdual3 = residual_block(in_channels=256, out_channels=128, t=2)
        self.up_resdual2 = residual_block(in_channels=128, out_channels=64, t=2)

        self.conv_1x1 = nn.Conv2d(64, num_classes + 1, kernel_size=1, stride=1, padding=0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self,x):
        rcnn1_out=self.residual_rcnn1(x)
        down1_out=self.maxpool(rcnn1_out)
        rcnn2_out=self.residual_rcnn2(down1_out)
        down2_out=self.maxpool(rcnn2_out)
        rcnn3_out=self.residual_rcnn3(down2_out)
        down3_out=self.maxpool(rcnn3_out)
        rcnn4_out=self.residual_rcnn4(down3_out)
        down4_out=self.maxpool(rcnn4_out)
        rcnn5_out=self.residual_rcnn5(down4_out)

        up5_out=self.up_resdual5(torch.cat((self.up5(rcnn5_out),rcnn4_out),dim=1))
        up4_out=self.up_resdual4(torch.cat((self.up4(up5_out),rcnn3_out),dim=1))
        up3_out=self.up_resdual3(torch.cat((self.up3(up4_out),rcnn2_out),dim=1))
        up2_out=self.up_resdual2(torch.cat((self.up2(up3_out),rcnn1_out),dim=1))
        up1_out=self.conv_1x1(up2_out)
        return up1_out




