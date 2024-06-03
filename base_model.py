import random
import torchvision.transforms as transforms
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
from PIL import Image

# 加入随机数使得结果可复现
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)  # 随机数

# 数据集类
class SegmentationDataset(Dataset):
    def __init__(self, image_list, label_list, transform=None, label_transform=None):
        self.image_list = image_list
        self.label_list = label_list
        self.transform = transform
        self.label_transform = label_transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = Image.open(self.image_list[idx])
        label = Image.open(self.label_list[idx])

        if self.transform:
            image = self.transform(image)

        if self.label_transform:
            label = self.label_transform(label)
        else:
            label = np.array(label)
            label = torch.as_tensor(label, dtype=torch.long)

        return image, label


# 读取文件路径函数
def read_paths_from_file(file_path):
    image_paths = []
    label_paths = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            image_path, label_path = line.strip().split()
            image_paths.append(image_path)
            label_paths.append(label_path)
    return image_paths, label_paths


# 读取路径信息
file_path = r'/mnt/workspace/ENet/train1.txt'
image_paths, label_paths = read_paths_from_file(file_path)

# 图像变换
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 调整大小
    transforms.ToTensor(),  # 转换为张量
])

# 标签变换
label_transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 调整大小
    transforms.Lambda(lambda img: torch.as_tensor(np.array(img), dtype=torch.long))  # 转换为张量
])

# 创建数据集
train_dataset = SegmentationDataset(image_paths, label_paths, transform=transform, label_transform=label_transform)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# 定义初始块
class InitialBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False, relu=True):
        super().__init__()
        activation = nn.ReLU if relu else nn.PReLU
        self.main_branch = nn.Conv2d(in_channels, out_channels - 3, kernel_size=3, stride=2, padding=1, bias=bias)
        self.ext_branch = nn.MaxPool2d(3, stride=2, padding=1)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.out_activation = activation()

    def forward(self, x):
        main = self.main_branch(x)
        ext = self.ext_branch(x)
        out = torch.cat((main, ext), 1)
        out = self.batch_norm(out)
        return self.out_activation(out)

# 定义常规瓶颈层
class RegularBottleneck(nn.Module):
    def __init__(self, channels, internal_ratio=4, kernel_size=3, padding=0, dilation=1, asymmetric=False,
                 dropout_prob=0, bias=False, relu=True):
        super().__init__()
        if internal_ratio <= 1 or internal_ratio > channels:
            raise RuntimeError(
                f"Value out of range. Expected value in the interval [1, {channels}], got internal_scale={internal_ratio}.")
        internal_channels = channels // internal_ratio
        activation = nn.ReLU if relu else nn.PReLU
        self.ext_conv1 = nn.Sequential(nn.Conv2d(channels, internal_channels, kernel_size=1, stride=1, bias=bias),
                                       nn.BatchNorm2d(internal_channels), activation())
        if asymmetric:
            self.ext_conv2 = nn.Sequential(
                nn.Conv2d(internal_channels, internal_channels, kernel_size=(kernel_size, 1), stride=1,
                          padding=(padding, 0), dilation=dilation, bias=bias), nn.BatchNorm2d(internal_channels),
                activation(),
                nn.Conv2d(internal_channels, internal_channels, kernel_size=(1, kernel_size), stride=1,
                          padding=(0, padding), dilation=dilation, bias=bias), nn.BatchNorm2d(internal_channels),
                activation()
            )
        else:
            self.ext_conv2 = nn.Sequential(
                nn.Conv2d(internal_channels, internal_channels, kernel_size=kernel_size, stride=1, padding=padding,
                          dilation=dilation, bias=bias), nn.BatchNorm2d(internal_channels), activation())
        self.ext_conv3 = nn.Sequential(nn.Conv2d(internal_channels, channels, kernel_size=1, stride=1, bias=bias),
                                       nn.BatchNorm2d(channels), activation())
        self.ext_regul = nn.Dropout2d(p=dropout_prob)
        self.out_activation = activation()

    def forward(self, x):
        main = x
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)
        out = main + ext
        return self.out_activation(out)

# 定义下采样瓶颈层
class DownsamplingBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, internal_ratio=4, return_indices=False, dropout_prob=0, bias=False,
                 relu=True):
        super().__init__()
        self.return_indices = return_indices
        if internal_ratio <= 1 or internal_ratio > in_channels:
            raise RuntimeError(
                f"Value out of range. Expected value in the interval [1, {in_channels}], got internal_scale={internal_ratio}.")
        internal_channels = in_channels // internal_ratio
        activation = nn.ReLU if relu else nn.PReLU
        self.main_max1 = nn.MaxPool2d(2, stride=2, return_indices=return_indices)
        self.ext_conv1 = nn.Sequential(nn.Conv2d(in_channels, internal_channels, kernel_size=2, stride=2, bias=bias),
                                       nn.BatchNorm2d(internal_channels), activation())
        self.ext_conv2 = nn.Sequential(
            nn.Conv2d(internal_channels, internal_channels, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.BatchNorm2d(internal_channels), activation())
        self.ext_conv3 = nn.Sequential(nn.Conv2d(internal_channels, out_channels, kernel_size=1, stride=1, bias=bias),
                                       nn.BatchNorm2d(out_channels), activation())
        self.ext_regul = nn.Dropout2d(p=dropout_prob)
        self.out_activation = activation()

    def forward(self, x):
        if self.return_indices:
            main, max_indices = self.main_max1(x)
        else:
            main = self.main_max1(x)
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)
        n, ch_ext, h, w = ext.size()
        ch_main = main.size()[1]
        padding = torch.zeros(n, ch_ext - ch_main, h, w)
        if main.is_cuda:
            padding = padding.cuda()
        main = torch.cat((main, padding), 1)
        out = main + ext
        return self.out_activation(out), max_indices

# 定义上采样瓶颈层
class UpsamplingBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, internal_ratio=4, dropout_prob=0, bias=False, relu=True):
        super().__init__()
        if internal_ratio <= 1 or internal_ratio > in_channels:
            raise RuntimeError(
                f"Value out of range. Expected value in the interval [1, {in_channels}], got internal_scale={internal_ratio}.")
        internal_channels = in_channels // internal_ratio
        activation = nn.ReLU if relu else nn.PReLU
        self.main_conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias),
                                        nn.BatchNorm2d(out_channels))
        self.main_unpool1 = nn.MaxUnpool2d(kernel_size=2)
        self.ext_conv1 = nn.Sequential(nn.Conv2d(in_channels, internal_channels, kernel_size=1, bias=bias),
                                       nn.BatchNorm2d(internal_channels), activation())
        self.ext_tconv1 = nn.ConvTranspose2d(internal_channels, internal_channels, kernel_size=2, stride=2, bias=bias)
        self.ext_tconv1_bnorm = nn.BatchNorm2d(internal_channels)
        self.ext_tconv1_activation = activation()
        self.ext_conv2 = nn.Sequential(nn.Conv2d(internal_channels, out_channels, kernel_size=1, bias=bias),
                                       nn.BatchNorm2d(out_channels))
        self.ext_regul = nn.Dropout2d(p=dropout_prob)
        self.out_activation = activation()

    def forward(self, x, max_indices, output_size):
        main = self.main_conv1(x)
        main = self.main_unpool1(main, max_indices, output_size=output_size)
        ext = self.ext_conv1(x)
        ext = self.ext_tconv1(ext, output_size=output_size)
        ext = self.ext_tconv1_bnorm(ext)
        ext = self.ext_tconv1_activation(ext)
        ext = self.ext_conv2(ext)
        ext = self.ext_regul(ext)
        out = main + ext
        return self.out_activation(out)

# 定义ENet模型
class ENet(nn.Module):
    def __init__(self, num_classes, encoder_relu=False, decoder_relu=True):
        super().__init__()
        self.initial_block = InitialBlock(3, 16, relu=encoder_relu)
        self.downsample1_0 = DownsamplingBottleneck(16, 64, return_indices=True, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_1 = RegularBottleneck(64, padding=1, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_2 = RegularBottleneck(64, padding=1, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_3 = RegularBottleneck(64, padding=1, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_4 = RegularBottleneck(64, padding=1, dropout_prob=0.01, relu=encoder_relu)
        self.downsample2_0 = DownsamplingBottleneck(64, 128, return_indices=True, dropout_prob=0.1, relu=encoder_relu)
        self.regular2_1 = RegularBottleneck(128, padding=1, dropout_prob=0.1, relu=encoder_relu)
        self.dilated2_2 = RegularBottleneck(128, dilation=2, padding=2, dropout_prob=0.1, relu=encoder_relu)
        self.asymmetric2_3 = RegularBottleneck(128, kernel_size=5, padding=2, asymmetric=True, dropout_prob=0.1,
                                               relu=encoder_relu)
        self.dilated2_4 = RegularBottleneck(128, dilation=4, padding=4, dropout_prob=0.1, relu=encoder_relu)
        self.regular2_5 = RegularBottleneck(128, padding=1, dropout_prob=0.1, relu=encoder_relu)
        self.dilated2_6 = RegularBottleneck(128, dilation=8, padding=8, dropout_prob=0.1, relu=encoder_relu)
        self.asymmetric2_7 = RegularBottleneck(128, kernel_size=5, padding=2, asymmetric=True, dropout_prob=0.1,
                                               relu=encoder_relu)
        self.dilated2_8 = RegularBottleneck(128, dilation=16, padding=16, dropout_prob=0.1, relu=encoder_relu)
        self.upsample4_0 = UpsamplingBottleneck(128, 64, dropout_prob=0.1, relu=decoder_relu)
        self.regular4_1 = RegularBottleneck(64, padding=1, dropout_prob=0.1, relu=decoder_relu)
        self.regular4_2 = RegularBottleneck(64, padding=1, dropout_prob=0.1, relu=decoder_relu)
        self.upsample5_0 = UpsamplingBottleneck(64, 16, dropout_prob=0.1, relu=decoder_relu)
        self.regular5_1 = RegularBottleneck(16, padding=1, dropout_prob=0.1, relu=decoder_relu)
        self.fullconv = nn.ConvTranspose2d(16, num_classes, kernel_size=3, stride=2, padding=1, output_padding=1)
    def forward(self, x):
        x = self.initial_block(x)
        x, max_indices1 = self.downsample1_0(x)
        x = self.regular1_1(x)
        x = self.regular1_2(x)
        x = self.regular1_3(x)
        x = self.regular1_4(x)
        x, max_indices2 = self.downsample2_0(x)
        x = self.regular2_1(x)
        x = self.dilated2_2(x)
        x = self.asymmetric2_3(x)
        x = self.dilated2_4(x)
        x = self.regular2_5(x)
        x = self.dilated2_6(x)
        x = self.asymmetric2_7(x)
        x = self.dilated2_8(x)
        x = self.upsample4_0(x, max_indices2, output_size=(64, 64))
        x = self.regular4_1(x)
        x = self.regular4_2(x)
        x = self.upsample5_0(x, max_indices1, output_size=(128, 128))
        x = self.regular5_1(x)
        x = self.fullconv(x)
        return x

# 计算像素准确率
def pixel_accuracy(output, target):
    _, preds = torch.max(output, 1)
    correct = (preds == target).float()
    acc = correct.sum() / correct.numel()
    return acc

# 模型实例化
num_classes = 12
model = ENet(num_classes)
# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练函数
def train_model(model, train_loader, criterion, optimizer, num_epochs=25, print_batches=3):
    best_accuracy = 0.0
    best_loss = float('inf')
    best_epoch = 0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_accuracy = 0.0
        batch_count = 0
        for inputs, labels in train_loader:
            if batch_count < print_batches:
                print(f"Epoch {epoch}, Batch {batch_count}, Inputs shape: {inputs.shape}, Labels shape: {labels.shape}")
            optimizer.zero_grad()
            outputs = model(inputs)
            if batch_count < print_batches:
                print(f"Epoch {epoch}, Batch {batch_count}, Outputs shape: {outputs.shape}")
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            running_accuracy += pixel_accuracy(outputs, labels) * inputs.size(0)
            batch_count += 1
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_accuracy = running_accuracy / len(train_loader.dataset)
        print(f'Epoch {epoch}/{num_epochs - 1}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')
        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            best_epoch = epoch
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_epoch = epoch
    
    print(f'Best Accuracy: {best_accuracy:.4f} at Epoch {best_epoch}, Best Loss: {best_loss:.4f}')

# 训练模型并打印形状信息
train_model(model, train_loader, criterion, optimizer, num_epochs=25, print_batches=3)
