import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class SegmentationDataset(Dataset):
    def __init__(self, image_list, label_list, transform=None):
        self.image_list = image_list
        self.label_list = label_list
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = Image.open(self.image_list[idx])
        label = Image.open(self.label_list[idx])

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        # 标签应为LongTensor类型，且不需要标准化
        label = torch.as_tensor(label, dtype=torch.long)

        return image, label


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
file_path =r'C:\Users\29918\Desktop\ENet\data\CamVid\train1.txt'
image_paths, label_paths = read_paths_from_file(file_path)

# 图像变换
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 调整大小
    transforms.ToTensor(),  # 转换为张量
])

# 创建数据集
dataset = SegmentationDataset(image_paths, label_paths, transform=transform)

# 创建数据加载器
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# 检查数据加载
for images, labels in dataloader:
    print(images.size(), labels.size())
    break
