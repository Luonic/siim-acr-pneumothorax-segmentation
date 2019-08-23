import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import hrnet
# https://github.com/lukemelas/EfficientNet-PyTorch
import efficientnet_pytorch


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

class Dropout2d(nn.Module):
    def __init__(self, p):
        super(Dropout2d, self).__init__()
        self.p = torch.tensor(1. - p)

    def forward(self, input):
        if self.training:
            probs = self.p.view(1, 1, 1, 1)
            probs = probs.repeat(1, input.size(1), 1, 1)
            dist = torch.distributions.bernoulli.Bernoulli(probs=probs)
            dropout_mask = dist.sample().type(input.type())
            # print(dropout_mask)
            return input * dropout_mask
        else:
            return input


class UnetBlock(nn.Module):
    def __init__(self, up_in, x_in, n_out, norm_layer=None, upsample=False):
        self.training = False
        self.dropout_prob = 0.1
        self.up_in = up_in
        self.x_in = x_in
        self.n_out = n_out
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.upsample = upsample
        up_out = x_out = n_out // 2
        if upsample:
            self.tr_conv = nn.Conv2d(up_in, up_out, 1)
        else:
            self.tr_conv = nn.ConvTranspose2d(up_in, up_out, 2, stride=2)
        self.x_conv = nn.Conv2d(x_in, x_out, 1)

        self.norm = norm_layer(n_out)
        self.cat_dropout = nn.Dropout2d(self.dropout_prob, inplace=False)

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(n_out, n_out, kernel_size=3, padding=1, bias=True),
            norm_layer(n_out),
            nn.ReLU(inplace=True),
            # nn.Dropout2d(self.dropout_prob)
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(n_out, n_out, kernel_size=3, padding=1, bias=True),
            norm_layer(n_out),
            nn.ReLU(inplace=True),
            # nn.Dropout2d(self.dropout_prob)
        )

    def forward(self, up_p, x_p):
        up_p = self.tr_conv(up_p)
        if self.upsample:
            up_p = F.interpolate(up_p, (x_p.size(2), x_p.size(3)), mode='nearest')
        x_p = self.x_conv(x_p)
        cat_p = torch.cat([up_p, x_p], dim=1)
        cat_p = self.norm(cat_p)
        cat_p = F.relu(cat_p, inplace=True)
        cat_p = self.cat_dropout(cat_p)

        out = self.conv_block_1(cat_p)
        out = self.conv_block_2(out)
        return out

class ResidualUnetBlock(nn.Module):
    def __init__(self, up_in, x_in, n_out, norm_layer=None, upsample=False):
        self.training = False
        self.dropout_prob = 0.1
        self.up_in = up_in
        self.x_in = x_in
        self.n_out = n_out
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.upsample = upsample
        up_out = x_out = n_out // 2
        if upsample:
            self.tr_conv = nn.Conv2d(up_in, up_out, 1)
        else:
            self.tr_conv = nn.ConvTranspose2d(up_in, up_out, 2, stride=2)
        self.x_conv = nn.Conv2d(x_in, x_out, 1)

        self.norm = norm_layer(n_out)
        self.cat_dropout = nn.Dropout2d(self.dropout_prob, inplace=False)

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(n_out, n_out, kernel_size=3, padding=1, bias=True),
            norm_layer(n_out),
            nn.ReLU(inplace=True),
            # nn.Dropout2d(self.dropout_prob)
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(n_out, n_out, kernel_size=3, padding=1, bias=True),
            norm_layer(n_out),
            # nn.Dropout2d(self.dropout_prob)
        )

    def forward(self, up_p, x_p):

        up_p = self.tr_conv(up_p)
        if self.upsample:
            up_p = F.interpolate(up_p, (x_p.size(2), x_p.size(3)), mode='nearest')
        x_p = self.x_conv(x_p)
        cat_p = torch.cat([up_p, x_p], dim=1)
        identity = cat_p = self.norm(cat_p)
        cat_p = F.relu(cat_p, inplace=True)
        cat_p = self.cat_dropout(cat_p)

        out = self.conv_block_1(cat_p)
        out = self.conv_block_2(out)
        out = identity + out
        out = F.relu(out)
        return out


class ResNetUNet(nn.Module):
    def __init__(self, n_classes, upsample=False):
        super(ResNetUNet, self).__init__()
        # norm_layer = lambda in_channels: nn.GroupNorm(8, in_channels) # for now 8 is best
        norm_layer = None
        self.backbone = torchvision.models.resnet34(zero_init_residual=True, pretrained=True, norm_layer=norm_layer)
        self.up1 = UnetBlock(512, 256, 256, upsample=upsample, norm_layer=norm_layer)
        self.up2 = UnetBlock(256, 128, 128, upsample=upsample, norm_layer=norm_layer)
        self.up3 = UnetBlock(128, 64, 64, upsample=upsample, norm_layer=norm_layer)
        self.up4 = UnetBlock(64, 64, 64, upsample=upsample, norm_layer=norm_layer)
        # self.up5 = nn.Sequential(
        #     nn.ConvTranspose2d(64, 32, 4, stride=2, bias=True),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),
        #     nn.Conv2d(32, n_classes, 3)
        # )
        self.up5 = nn.Sequential(
            nn.Dropout2d(p=0.2),
            nn.Conv2d(64, n_classes, 3)
        )

        self.ada_p = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout2d(p=0.4),
            nn.Conv2d(512, n_classes, 1))
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

        # x1 = F.relu(x1)
        # x2 = F.relu(x2)
        # x3 = F.relu(x3)
        # x4 = F.relu(x4)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, conv1)
        x = self.up5(x)

        mask_logits = F.interpolate(x, (inp.shape[2], inp.shape[3]), mode='bilinear')
        mask_probs = self.sigmoid(mask_logits)

        cls = self.ada_p(x4)
        cls_logits = self.classifier(cls)
        cls_logits = cls_logits.view((-1, 1))
        cls_probs = torch.sigmoid(cls_logits)

        out = {'mask': mask_probs, 'mask_logits': mask_logits,
               'class': cls_probs, 'class_logits': cls_logits}
        return out


class UNetPlusPlus(nn.Module):
    class UnetPlusPlusBlock(nn.Module):
        def __init__(self, up_in, x_in, norm_layer=None):
            self.dropout_prob = 0.5
            self.up_in = up_in
            self.x_in = x_in
            self.n_out = x_in
            super().__init__()
            if norm_layer is None:
                norm_layer = nn.BatchNorm2d
            up_out = x_out = self.n_out // 2
            self.tr_conv = nn.ConvTranspose2d(up_in, up_out, 2, stride=2)
            self.x_conv = nn.Conv2d(x_in, x_out, 1)
            self.norm = norm_layer(self.n_out)

            self.conv_block_1 = nn.Sequential(
                nn.Conv2d(self.n_out, self.n_out, kernel_size=3, padding=1, bias=False),
                norm_layer(self.n_out),
                nn.ReLU(inplace=True),
            )

            self.conv_block_2 = nn.Sequential(
                nn.Conv2d(self.n_out, self.n_out, kernel_size=3, padding=1, bias=False),
                norm_layer(self.n_out),
                nn.ReLU(inplace=True),
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

    def __init__(self, num_classes, num_backbone_channels, norm_layer=None):
        super(UNetPlusPlus, self).__init__()
        self.num_backbone_channels = num_backbone_channels
        self.stages = []
        for stage_idx in range(1, len(num_backbone_channels)):
            stage = []
            for resolution_idx in range(1, len(num_backbone_channels) - (stage_idx - 1)):
                stage.append(self.UnetPlusPlusBlock(
                    x_in=num_backbone_channels[resolution_idx - 1],
                    up_in=num_backbone_channels[resolution_idx]
                ))
            self.stages.append(stage)

        self.classifier = nn.Conv2d(in_channels=self.num_backbone_channels[0], out_channels=num_classes, kernel_size=1)

    def forward(self, inputs):
        # Inputs - iterable of different resolution featuremaps
        stages_featuremaps = [inputs]
        for stage_idx in range(1, len(self.num_backbone_channels)):
            stage_featuremaps = []
            for resolution_idx in range(1, len(self.num_backbone_channels) - (stage_idx - 1)):
                x_in = stages_featuremaps[stage_idx - 1][resolution_idx - 1]
                up_in = stages_featuremaps[stage_idx - 1][resolution_idx]
                stage_featuremaps.append(self.stages[stage_idx - 1][resolution_idx - 1](up_in, x_in))
            stages_featuremaps.append(stage_featuremaps)

        mask_levels = [stage_featuremaps[0] for stage_featuremaps in stages_featuremaps]

        return mask_levels


class ResNetUNetPlusPlus(nn.Module):
    def __init__(self, n_classes):
        super(ResNetUNetPlusPlus, self).__init__()
        self.backbone = torchvision.models.resnet34(zero_init_residual=True, pretrained=True)
        self.unetplusplus = UNetPlusPlus(num_classes=n_classes, num_backbone_channels=(64, 64, 128, 256, 512))

        self.upsample_and_classify = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.Dropout2d(p=0.1),
            nn.Conv2d(32, n_classes, 3)
        )

        self.ada_p = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            # nn.Dropout2d(p=0.5),
            nn.Conv2d(512, n_classes, 1))
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

        cls = self.ada_p(x4)
        cls_logits = self.classifier(cls)
        cls_logits = cls_logits.view((-1, 1))
        cls_probs = torch.sigmoid(cls_logits)

        mask_levels = self.unetplusplus((conv1, x1, x2, x3, x4))
        mask_levels = [self.upsample_and_classify(mask_level) for mask_level in mask_levels]

        avg_mask = [torch.unsqueeze(mask_level, dim=0) for mask_level in mask_levels]
        avg_mask = torch.cat(avg_mask, dim=0)
        mask_logits = avg_mask.mean(dim=0, keepdim=False)
        mask_logits = F.interpolate(mask_logits, (inp.shape[2], inp.shape[3]), mode='bilinear')
        mask_probs = self.sigmoid(mask_logits)
        out = {'mask': mask_probs, 'mask_logits': mask_logits, 'mask_levels_logits': mask_levels,
               'class': cls_probs, 'class_logits': cls_logits}
        return out


class EfficientUNet(nn.Module):
    def __init__(self, n_classes):
        super(EfficientUNet, self).__init__()
        self.backbone = efficientnet_pytorch.EfficientNet.from_pretrained('efficientnet-b0')
        # b7
        # self.feature_idxs = [(4, 32),  # / 2
        #                      (11, 48),  # / 4
        #                      (18, 80),  # / 8
        #                      (38, 224),  # / 16
        #                      (55, 640)]  # / 32

        # b5
        # self.feature_idxs = [(3, 24),  # / 2
        #                      (8, 40),  # / 4
        #                      (13, 64),  # / 8
        #                      (27, 176),  # / 16
        #                      (39, 512)]  # / 32

        self.feature_idxs = [(1, 16),  # / 2
                             (3, 24),  # / 4
                             (5, 40),  # / 8
                             (11, 112),  # / 16
                             (16, 320)]  # / 32

        self.up1 = UnetBlock(320, 112, 112)
        self.up2 = UnetBlock(112, 40, 40)
        self.up3 = UnetBlock(40, 24, 24)
        self.up4 = UnetBlock(24, 16, 16)
        self.up5 = nn.Sequential(nn.ConvTranspose2d(16, 16, 4, stride=2, bias=False),
                                 nn.BatchNorm2d(16),
                                 nn.ReLU(),
                                 nn.Conv2d(16, n_classes, 3))

        self.ada_p = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Conv2d(320, n_classes, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inp):
        features = []
        # Stem
        x = F.relu(self.backbone._bn0(self.backbone._conv_stem(inp)))
        features.append(x)

        # Blocks
        for idx, block in enumerate(self.backbone._blocks):
            drop_connect_rate = self.backbone._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.backbone._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            features.append(x)

        # for i, tensor in enumerate(features):
        #     print(i, tensor.shape)

        x1 = F.relu(features[self.feature_idxs[0][0]])
        x2 = F.relu(features[self.feature_idxs[1][0]])
        x3 = F.relu(features[self.feature_idxs[2][0]])
        x4 = F.relu(features[self.feature_idxs[3][0]])
        x5 = F.relu(features[self.feature_idxs[4][0]])

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.up5(x)

        mask_logits = F.interpolate(x, (inp.shape[2], inp.shape[3]), mode='bilinear')
        mask_probs = self.sigmoid(mask_logits)

        cls = self.ada_p(x5)
        cls_logits = self.classifier(cls)
        cls_logits = cls_logits.view((-1, 1))
        cls_probs = torch.sigmoid(cls_logits)

        out = {'mask': mask_probs, 'mask_logits': mask_logits,
               'class': cls_probs, 'class_logits': cls_logits}
        return out
