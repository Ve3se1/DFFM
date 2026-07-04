import torch
import torch.nn as nn
import torchvision.models as models


# 定义空间自适应权重模块（SpatialWeightedMap）
class SpatialWeightedMap(nn.Module):
    def __init__(self, channels):
        super(SpatialWeightedMap, self).__init__()
        self.weight_generator = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        spatial_weight = self.weight_generator(x)
        return spatial_weight * x


# 定义DFFM模块
class DFFMBlock(nn.Module):
    def __init__(self, channels):
        super(DFFMBlock, self).__init__()
        self.dominant_eye = SpatialWeightedMap(channels)
        self.conv_color = nn.Conv2d(channels, channels, 3, padding=1)
        self.color_fusion_ratio = nn.Parameter(torch.tensor(0.5))

    def forward(self, x_rgb, x_lab_enhanced):
        dominant_features = self.dominant_eye(x_rgb)
        color_features = self.conv_color(x_lab_enhanced)

        color_ratio = torch.sigmoid(self.color_fusion_ratio)
        fused_color = color_ratio * color_features + (1 - color_ratio) * x_rgb

        fused_features = dominant_features + fused_color
        return fused_features


# 修改后的ResNet Block，将DFFM融合到其中
class ResNetDFFM(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNetDFFM, self).__init__()
        resnet = models.resnet50(pretrained=True)

        # 使用ResNet原始stem层
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        # 在layer1前插入DFFM Block
        self.dffm_block = DFFMBlock(channels=64)

        # ResNet后续层
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x_rgb, x_lab_enhanced):
        # 原始stem层
        x_rgb = self.conv1(x_rgb)
        x_rgb = self.bn1(x_rgb)
        x_rgb = self.relu(x_rgb)
        x_rgb = self.maxpool(x_rgb)

        # Lab增强图像经过stem层（复用stem）
        x_lab = self.conv1(x_lab_enhanced)
        x_lab = self.bn1(x_lab)
        x_lab = self.relu(x_lab)
        x_lab = self.maxpool(x_lab)

        # DFFM Block融合
        x = self.dffm_block(x_rgb, x_lab)

        # 后续ResNet层
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


# 测试代码
if __name__ == "__main__":
    model = ResNetDFFM(num_classes=101)
    x_rgb = torch.randn(4, 3, 224, 224)  # 模拟RGB输入
    x_lab = torch.randn(4, 3, 224, 224)  # 模拟Lab增强输入
    output = model(x_rgb, x_lab)
    print(output.shape)  # 应输出 torch.Size([4, 101])
