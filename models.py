import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import hrnet


class ClassifierReductionBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ClassifierReductionBlock, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1)
        self.norm = nn.GroupNorm(16, out_ch)
        self.mp = nn.MaxPool2d(3, stride=2, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x1, x2):
        x = self.conv(x1)
        x = self.norm(x)
        x = self.mp(x)
        x = self.act(x)
        return x


class ClassifierMultiscaleBlock(nn.Module):
    def __init__(self, in_ch1, in_ch2):
        super(ClassifierMultiscaleBlock, self).__init__()
        self.conv = nn.Conv2d(in_ch1, in_ch2, 3, stride=1, padding=1)
        self.norm = nn.GroupNorm(16, in_ch2)
        self.mp = nn.MaxPool2d(3, stride=2, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x1, x2):
        x = self.conv(x1)
        x = self.norm(x)
        x = self.mp(x)
        x = self.act(x)
        x = x + x2
        return x


class Classifier(nn.Module):
    def __init__(self, num_scales=8, in_channels=[32, 64, 128, 256], max_channels=32, num_classes=1):
        super(Classifier, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.multiscale_layers = []
        self.layers = []
        for i in range(len(in_channels) - 1):
            c_in = in_channels[i]
            c_out = in_channels[i + 1]
            self.multiscale_layers.append(ClassifierMultiscaleBlock(c_in, c_out))

        for i in range(num_scales - len(in_channels)):
            c_in = in_channels[-1]
            c_out = min(max_channels, c_in * 2)
            self.layers.append(ClassifierReductionBlock(c_in, c_out))
            c_in = c_out

        self.multiscale_layers = nn.ModuleList(self.multiscale_layers)

        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.adaptive_max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 1))
        self.class_conv = nn.Conv2d(c_out * 2, num_classes, kernel_size=1, padding=0)

    def forward(self, input_list):
        x = input_list[0]
        for i, layer in enumerate(self.multiscale_layers):
            x = layer(x, input_list[i + 1])

        x1 = self.adaptive_avg_pool(x)
        x2 = self.adaptive_max_pool(x)
        x = torch.cat((x1, x2), dim=1)
        x = self.class_conv(x)
        x = F.sigmoid(x).view(-1, self.num_classes)
        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
            # elif isinstance(m, InPlaceABNSync):
            #     nn.init.constant_(m.weight, 1)
            #     nn.init.constant_(m.bias, 0)
            # elif isinstance(m, nn.GroupNorm):
            #     nn.init.constant_(m.weight, 1)
            #     nn.init.constant_(m.bias, 0)


class HRNetWithClassifier(nn.Module):
    def __init__(self):
        super(HRNetWithClassifier, self).__init__()
        self.hrnet = hrnet.HighResolutionNet(out_channels=1)
        self.hrnet.init_weights()
        self.classifier = Classifier(num_scales=8, in_channels=[32, 64, 128, 256], max_channels=256, num_classes=1)
        self.classifier.init_weights()

    def forward(self, inp):
        pred_dict = self.hrnet(inp)
        pred_dict['class'] = self.classifier(pred_dict['multiscale_features'])
        return pred_dict


class ModelWithLoss(nn.Module):
    def __init__(self, model, criterion):
        super(ModelWithLoss, self).__init__()
        self.model = model
        self.criterion = criterion

    def forward(self, input, targets):
        pred = self.model(input)
        loss = self.criterion(pred, targets)
        return pred, loss


class MyModel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, out_channels, 3, stride=1, padding=1, bias=False)
        self.norm = nn.GroupNorm(32, 1, )
        self.act = nn.ReLU()
        self.ada_mp = nn.AdaptiveMaxPool2d((1, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x = self.conv1(input)
        # x = self.norm(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.sigmoid(x)
        cls = self.ada_mp(x).view(-1, 1)
        out = {'mask': x, 'class': cls}
        return out


class MyResNetModel(nn.Module):
    def __init__(self):
        super(MyResNetModel, self).__init__()
        norm_layer = lambda in_planes: nn.GroupNorm(32, in_planes)
        self.backbone = torchvision.models.resnet34(zero_init_residual=False, norm_layer=norm_layer)
        self.out_conv1 = nn.Conv2d(448, 32, 3, stride=1, padding=1)  # 960
        self.out_conv2 = nn.Conv2d(32, 1, 3, stride=1, padding=1)
        self.ada_mp = nn.AdaptiveMaxPool2d((1, 1))
        self.sigmoid = nn.Sigmoid()
        self.bn_out = norm_layer(32)
        self.act = nn.ReLU()

        self.up5 = up(6, 5)
        self.up4 = up(4, 3)
        self.up3 = up(2, 1)

    # def forward(self, inp):
    #     x = self.backbone.conv1(inp)
    #     x = self.backbone.bn1(x)
    #     x = self.backbone.relu(x)
    #     x = self.backbone.maxpool(x)
    #     x = x1 = self.backbone.layer1(x)
    #     x = x2 = self.backbone.layer2(x)
    #     x = x3 = self.backbone.layer3(x)
    #     x = x4 = self.backbone.layer4(x)
    #
    #     x1 = F.interpolate(x1, (x1.shape[2], x1.shape[3]), mode='bilinear')
    #     x2 = F.interpolate(x2, (x1.shape[2], x1.shape[3]), mode='bilinear')
    #     x3 = F.interpolate(x3, (x1.shape[2], x1.shape[3]), mode='bilinear')
    #     x4 = F.interpolate(x4, (x1.shape[2], x1.shape[3]), mode='bilinear')
    #
    #     x = torch.cat([x1, x2, x3], dim=1)
    #     x = self.out_conv1(x)
    #     x = self.bn_out(x)
    #     x = self.act(x)
    #     x = F.interpolate(x, (inp.shape[2], inp.shape[3]), mode='bilinear')
    #     x = self.out_conv2(x)
    #     x = self.sigmoid(x)
    #     cls = self.ada_mp(x).view(-1, 1)
    #     out = {'mask': x, 'class': cls}
    #     return out

    def forward(self, inp):
        x = self.backbone.conv1(inp)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = x1 = self.backbone.layer1(x)
        x = x2 = self.backbone.layer2(x)
        x = x3 = self.backbone.layer3(x)
        x = x4 = self.backbone.layer4(x)

        x = self.up4(x4, x3, )
        x = self.up3(x, x2)
        x = self.up2(x, x1)
        x = F.interpolate(x, (inp.shape[2], inp.shape[3]), mode='bilinear')
        # x = self.up2(x, x2)
        # x = self.up1(x, x1)
        x = self.outc(x)
        x = self.sigmoid(x)
        cls = self.ada_mp(x).view(-1, 1)
        out = {'mask': x, 'class': cls}
        return out


class MyMobileNetModel(nn.Module):
    def __init__(self):
        super(MyMobileNetModel, self).__init__()
        self.out_conv1 = nn.Conv2d(512, 32, 3, stride=1, padding=1)  # 960
        self.out_conv2 = nn.Conv2d(32, 1, 3, stride=1, padding=1)
        self.ada_mp = nn.AdaptiveMaxPool2d((1, 1))
        self.sigmoid = nn.Sigmoid()
        self.bn_out = nn.BatchNorm2d(32)
        self.act = nn.ReLU()

    def forward(self, inp):
        x = self.backbone.features(inp)

        x = F.interpolate(x, (inp.shape[2], inp.shape[3]))

        x = self.out_conv1(x)
        x = self.bn_out(x)
        x = self.act(x)
        x = self.out_conv2(x)
        x = self.sigmoid(x)
        cls = self.ada_mp(x).view(-1, 1)
        out = {'mask': x, 'class': cls}
        return out


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch, norm_layer=nn.BatchNorm2d):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            norm_layer(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            norm_layer(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, up_in_ch, in_ch, out_ch, norm_layer, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(up_in_ch, up_in_ch, 2, stride=2)

        self.conv = double_conv(up_in_ch + in_ch, out_ch, norm_layer=norm_layer)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class UNet(nn.Module):

    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        c = 32
        self.inc = inconv(n_channels, c)
        self.down1 = down(c, c * 2)
        self.down2 = down(c * 2, c * 4)
        self.down3 = down(c * 4, c * 8)
        self.down4 = down(c * 8, c * 8)
        self.down5 = down(c * 8, c * 8)
        self.up5 = up(c * 16, c * 8)
        self.up4 = up(c * 16, c * 4)
        self.up3 = up(c * 8, c * 2)
        self.up2 = up(c * 4, c)
        self.up1 = up(c * 2, c)
        self.outc = outconv(c, n_classes)
        self.ada_mp = nn.AdaptiveMaxPool2d((1, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)

        x = self.up5(x6, x5)
        x = self.up4(x, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        x = self.outc(x)
        x = self.sigmoid(x)
        cls = self.ada_mp(x).view(-1, 1)
        out = {'mask': x, 'class': cls}
        return out


# class UnetBlock(nn.Module):
#     def __init__(self, up_in, x_in, n_out):
#         super().__init__()
#         up_out = x_out = n_out // 2
#         self.x_conv = nn.Conv2d(x_in, x_out, 1)
#         self.tr_conv = nn.ConvTranspose2d(up_in, up_out, 2, stride=2)
#         self.bn = nn.BatchNorm2d(n_out)
#
#     def forward(self, up_p, x_p):
#         up_p = self.tr_conv(up_p)
#         x_p = self.x_conv(x_p)
#         cat_p = torch.cat([up_p, x_p], dim=1)
#         return self.bn(F.relu(cat_p))

class UnetBlock(nn.Module):
    def __init__(self, up_in, x_in, n_out, norm_layer=None):
        self.up_in = up_in
        self.x_in = x_in
        self.n_out = n_out
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        up_out = x_out = n_out // 2
        self.tr_conv = nn.ConvTranspose2d(up_in, up_out, 2, stride=2)
        self.x_conv = nn.Conv2d(x_in, x_out, 1)

        self.norm = norm_layer(n_out)

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(n_out, n_out, kernel_size=3, padding=1, bias=False),
            norm_layer(n_out),
            nn.ReLU(inplace=True)
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(n_out, n_out, kernel_size=3, padding=1, bias=False),
            norm_layer(n_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, up_p, x_p):
        up_p = self.tr_conv(up_p)
        x_p = self.x_conv(x_p)
        cat_p = torch.cat([up_p, x_p], dim=1)
        cat_p = self.norm(cat_p)
        cat_p = F.relu(cat_p, inplace=True)

        out = self.conv_block_1(cat_p)
        out = self.conv_block_2(out)
        return out


class ResNetUNet(nn.Module):

    # def __init__(self, n_classes):
    #     super(ResNetUNet, self).__init__()
    #     bilinear_upsample = True
    #     # norm_layer = nn.BatchNorm2d
    #     # norm_layer = lambda in_planes: nn.GroupNorm(32, in_planes)
    #     norm_layer = None
    #     self.backbone = torchvision.models.resnet18(zero_init_residual=True, pretrained=True, norm_layer=norm_layer)
    #     norm_layer = nn.BatchNorm2d
    #     # norm_layer = lambda in_planes: nn.GroupNorm(32, in_planes)
    #     self.up4 = up(512, 256, 256, norm_layer=norm_layer, bilinear=bilinear_upsample)
    #     self.up3 = up(256, 128, 128, norm_layer=norm_layer, bilinear=bilinear_upsample)
    #     self.up2 = up(128, 64, 64, norm_layer=norm_layer, bilinear=bilinear_upsample)
    #     # self.up1 = up(64, 32, bilinear=bilinear_upsample)
    #     self.outc = outconv(64, n_classes)
    #     self.ada_mp = nn.AdaptiveMaxPool2d((1, 1))
    #     self.cls_conv = nn.Conv2d(512, n_classes, 1)
    #     self.sigmoid = nn.Sigmoid()

    def __init__(self, n_classes):
        super(ResNetUNet, self).__init__()
        self.backbone = torchvision.models.resnet34(zero_init_residual=True, pretrained=True)
        self.up1 = UnetBlock(512, 256, 256)
        self.up2 = UnetBlock(256, 128, 128)
        self.up3 = UnetBlock(128, 64, 64)
        self.up4 = UnetBlock(64, 64, 64)
        self.up5 = nn.Sequential(nn.ConvTranspose2d(64, 32, 4, stride=2, bias=False),
                                 nn.BatchNorm2d(32),
                                 nn.ReLU())

        self.outc = nn.Conv2d(32, n_classes, 3)
        self.ada_p = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(nn.Conv2d(512, 256, 1),
                                        nn.BatchNorm2d(256),
                                        nn.ReLU(inplace=False),
                                        nn.Conv2d(256, n_classes, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, inp):
        x = conv1 = self.backbone.conv1(inp)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = x1 = self.backbone.layer1(x)
        x = x2 = self.backbone.layer2(x)
        x = x3 = self.backbone.layer3(x)
        x = x4 = self.backbone.layer4(x)

        x1 = F.relu(x1)
        x2 = F.relu(x2)
        x3 = F.relu(x3)
        x4 = F.relu(x4)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, conv1)
        x = self.up5(x)

        # x = self.up1(x1, x1)
        x = self.outc(x)
        mask_logits = F.interpolate(x, (inp.shape[2], inp.shape[3]), mode='bilinear')
        mask_probs = self.sigmoid(mask_logits)

        cls = self.ada_p(x4)
        cls_logits = self.classifier(cls)
        cls_logits = cls_logits.view((-1, 1))
        cls_probs = torch.sigmoid(cls_logits)

        out = {'mask': mask_probs, 'mask_logits': mask_logits,
               'class': cls_probs, 'class_logits': cls_logits}
        return out