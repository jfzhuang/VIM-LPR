import torch
from torch import nn
from .resnet import resnet

adNum = 34


class ConvBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        x = self.conv1(input)
        return self.relu(self.bn(x))


class Spatial_path(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.convblock1 = ConvBlock(in_channels=3, out_channels=64)
        self.convblock2 = ConvBlock(in_channels=64, out_channels=128)
        self.convblock3 = ConvBlock(in_channels=128, out_channels=256)

    def forward(self, input):
        x = self.convblock1(input)
        x = self.convblock2(x)
        x = self.convblock3(x)
        return x


class AttentionRefinementModule(torch.nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()
        self.in_channels = in_channels
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, input):
        # global average pooling
        x = self.avgpool(input)
        assert self.in_channels == x.size(1), 'in_channels and out_channels should all be {}'.format(x.size(1))
        x = self.conv(x)
        x = self.sigmoid(self.bn(x))
        # channels of input and x should be same
        x = torch.mul(input, x)
        return x


class FeatureFusionModule(torch.nn.Module):

    def __init__(self, num_classes, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.convblock = nn.Sequential(
            ConvBlock(in_channels=self.in_channels, out_channels=512, stride=1),
            ConvBlock(in_channels=512, out_channels=num_classes, stride=1)
            )
        self.conv1 = nn.Conv2d(num_classes, num_classes, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_classes, num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, input_1, input_2):
        x = torch.cat((input_1, input_2), dim=1)
        assert self.in_channels == x.size(1), 'in_channels of ConvBlock should be {}'.format(x.size(1))
        feature = self.convblock(x)
        x = self.avgpool(feature)
        x = self.relu(self.conv1(x))
        x = self.sigmoid(self.conv2(x))
        x = torch.mul(feature, x)
        x = torch.add(x, feature)
        return x


class FeatureFusionModule_pos(torch.nn.Module):

    def __init__(self, num_classes, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.convblock = ConvBlock(in_channels=self.in_channels, out_channels=num_classes, stride=1)
        self.conv1 = nn.Conv2d(num_classes, num_classes, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(num_classes, num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, input_1, input_2):
        x = torch.cat((input_1, input_2), dim=1)
        assert self.in_channels == x.size(1), 'in_channels of ConvBlock should be {}'.format(x.size(1))
        feature = self.convblock(x)
        x = self.avgpool(feature)
        x = self.relu(self.conv1(x))
        x = self.sigmoid(self.conv2(x))
        x = torch.mul(feature, x)
        x = torch.add(x, feature)
        return x


class BiSeNet(torch.nn.Module):

    def __init__(self, num_classes, num_char, context_path):
        super().__init__()
        # build spatial path
        self.saptial_path = Spatial_path()

        # build context path
        self.context_path = resnet(context_path)

        # build attention refinement module
        if context_path == 'resnet101' or context_path == 'resnet50':
            self.attention_refinement_module1 = AttentionRefinementModule(1024, 1024)
            self.attention_refinement_module2 = AttentionRefinementModule(2048, 2048)
            self.feature_fusion_module1 = FeatureFusionModule(num_classes, 3328)
            self.feature_fusion_module2 = FeatureFusionModule_pos(num_char, 3328)
        else:
            self.attention_refinement_module1 = AttentionRefinementModule(256, 256)
            self.attention_refinement_module2 = AttentionRefinementModule(512, 512)
            self.feature_fusion_module1 = FeatureFusionModule(num_classes, 1024)
            self.feature_fusion_module2 = FeatureFusionModule_pos(num_char, 1024)
        # build final convolution
        self.conv_seg = nn.Conv2d(in_channels=num_classes, out_channels=num_classes, kernel_size=1)
        self.conv_pos = nn.Conv2d(in_channels=num_char, out_channels=num_char, kernel_size=1)
        self.bnSeg = nn.BatchNorm2d(num_classes)
        self.convblock = nn.Sequential(
            ConvBlock(in_channels=num_classes-1, out_channels=1000, kernel_size=5, stride=4, padding=2),
            nn.AdaptiveAvgPool2d((1, 1)),
            )
        self.classifier = nn.Linear(1000, adNum)

    def forward(self, input, label=None):
        # output of spatial path
        sx = self.saptial_path(input)
        # output of context path
        cx1, cx2, tail = self.context_path(input)
        cx1 = self.attention_refinement_module1(cx1)
        cx2 = self.attention_refinement_module2(cx2)
        cx2 = torch.mul(cx2, tail)
        # upsampling
        cx1 = torch.nn.functional.upsample(cx1, size=(sx.shape[2], sx.shape[3]), mode='bilinear', align_corners=True)
        cx2 = torch.nn.functional.upsample(cx2, size=(sx.shape[2], sx.shape[3]), mode='bilinear', align_corners=True)
        cx = torch.cat((cx1, cx2), dim=1)

        # output of feature fusion module
        segmap = self.feature_fusion_module1(sx, cx)
        posmap = self.feature_fusion_module2(sx, cx)
        segmapp = self.conv_seg(segmap)
        posmapp = self.conv_pos(posmap)
        # segmap = torch.nn.functional.upsample(segmapp, size=(input.shape[2], input.shape[3]), mode='bilinear', align_corners=True)
        segmapp = self.bnSeg(segmapp)
        posmapp = nn.functional.softmax(posmapp, dim=1)
        # posmap = torch.nn.functional.upsample(posmapp, size=(input.shape[2], input.shape[3]), mode='bilinear', align_corners=True)

        out = []
        for i in range(posmapp.size()[1] - 1):
            seg_pos = segmapp.narrow(1, 0, segmapp.size()[1]-1).mul(posmapp.narrow(1, i, 1))
            out.append(seg_pos)
        out = torch.cat(out, 0)
        cls_feat = self.convblock(out)

        y = self.classifier(cls_feat.view(cls_feat.size()[0], -1))
        batch = segmapp.size()[0]
        y0 = y.narrow(0, 0, batch)
        y1 = y.narrow(0, 1*batch, batch)
        y2 = y.narrow(0, 2*batch, batch)
        y3 = y.narrow(0, 3*batch, batch)
        y4 = y.narrow(0, 4*batch, batch)
        y5 = y.narrow(0, 5*batch, batch)

        return [y0, y1, y2, y3, y4, y5]


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    model = BiSeNet(num_classes=35, num_char=7, context_path='resnet34')
    model = nn.DataParallel(model)

    model = model.cuda()
    for name, key in model.named_parameters():
        print(name)
    x = torch.rand(2, 3, 50, 160)
    y = model(x)
