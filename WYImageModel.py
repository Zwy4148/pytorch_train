import datetime
import os

import dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import ToTensor, Normalize, RandomCrop
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 启动cuda，如果能用的话
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    print("start cuda")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 创建一个在CUDA设备上的生成器
    generator = torch.Generator(device=device)


# 图像模型
class WYImageModel(nn.Module):
    def __init__(self):
        super(WYImageModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=5, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=5, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(12)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(24)
        self.conv5 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=5, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(24)
        self.fc1 = nn.Linear(24 * 106 * 106, 10)

    # 前向传播
    def forward(self, input):
        output = F.relu(self.bn1(self.conv1(input)))
        output = F.relu(self.bn2(self.conv2(output)))
        output = self.pool(output)
        output = F.relu(self.bn4(self.conv4(output)))
        output = F.relu(self.bn5(self.conv5(output)))
        output = output.view(-1, 24 * 106 * 106)
        output = self.fc1(output)

        return output


# 数据集
class WYImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transforms = transform
        self.classes = sorted(os.listdir(root_dir))
        self.img_path = []
        self.labels = []

        for idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.root_dir, class_name)
            for image_name in os.listdir(class_dir):
                image_path = os.path.join(class_dir, image_name)
                self.img_path.append(image_path)
                self.labels.append(idx)

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, item):
        img_path = self.img_path[item]
        label = self.labels[item]
        image = Image.open(img_path).convert("RGB")
        if self.transforms:
            image = self.transforms(image)
        return image, label


# 训练工具
class TrainTools:
    def __init__(self, model: WYImageModel, dataloader: WYImageDataset):
        self.model = model
        self.dataloader = dataloader

        # 损失函数
        self.loss_fn = nn.CrossEntropyLoss()
        # 优化器
        self.optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    def save_model(self):
        tags = {}
        for index,tag in enumerate(self.dataloader):
            tags[index] = tag
        now_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = f"./WYImageModel_{now_time}.ditc"

        torch.save({"model":self.model.state_dict(),
                        "tags":tags}, path)

    def test_accuracy(self):
        self.model.eval()
        accuracy = 0.0
        total = 0.0
        with torch.no_grad():
            for data in tqdm(self.dataloader):
                images, tags = data
                output = self.model(images.to(device))
                image, predicted_tags = torch.max(output.data, 1)
                total += tags.size(0)
                accuracy += (predicted_tags == tags).sum().item()
        accuracy = (100 * accuracy / total)
        return accuracy

    def start_train(self, num_epochs):
        best_accuracy = 0.0
        accuracy_count = 0
        for epoch in tqdm(range(num_epochs)):
            running_loss = 0.0
            running_acc = 0.0
            for index, (images, tags) in enumerate(self.dataloader, 0):
                self.optimizer.zero_grad()
                output = self.model(images.to(device))
                loss = self.loss_fn(output, tags)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                if index % 10 == 9:
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, index + 1, running_loss / 10))
                    running_loss = 0.0
            accuracy = self.test_accuracy()
            print('For epoch', epoch + 1, 'the test accuracy over the whole test set is %d %%' % (accuracy))
            if accuracy > best_accuracy:
                if accuracy == 100:
                    accuracy_count += 1
                self.save_model()
                best_accuracy = accuracy
            if accuracy_count == 5:
                print("训练五次结果都为100%，暂时退出训练模式")
                break


def history_test():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    wy_dataset = WYImageDataset(r"E:\test_model\ImageDataset", transform)
    wy_dataloader = DataLoader(dataset=wy_dataset, batch_size=64, generator=generator)
    wy_model = WYImageModel().to(device)
    wy_train = TrainTools(wy_model, wy_dataloader)
    wy_train.start_train(100)
    wy_train.save_model()


def test_result(model_path, photo_path):
    result_change = {
        0: "Furina",
        1: "Xiangling"
    }
    # 定义模型结构
    model = WYImageModel()
    # 加载模型参数
    model.load_state_dict(torch.load(model_path))
    # 如果使用GPU，将模型移动到GPU上
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 定义图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 打开图像
    image = Image.open(photo_path)
    # 应用预处理
    image_tensor = transform(image).unsqueeze(0)
    # 如果使用GPU，将张量移动到GPU上
    image_tensor = image_tensor.to(device)

    # 预测结果
    # 将模型设置为评估模式
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output.data, 1)
    print(f"Predicted class: {result_change[predicted.item()]}")


if __name__ == '__main__':
    test_result(r"E:\test_model\WYImageModel_2024-12-24_00-49-15.pth",
                r"E:\test_model\data_save\Furina\124837067_p2_qg_フリーナ.jpg")
