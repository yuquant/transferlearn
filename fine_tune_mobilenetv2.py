# -*- coding: utf-8 -*-
"""
Author : Jason
Github : https://github.com/yuquant
Description : 
"""
import torch
from MobileNetV2 import MobileNetV2
from torchvision import datasets, transforms

# -*- coding: utf-8 -*-
"""
Author : Jason
Github : https://github.com/yuquant
Description :
卷积神经网络  迁移学习  vgg16 预训练权重训练
"""


import torch
import torchvision
from torchvision import datasets, transforms, models
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import time

torch.manual_seed(551)
torch.cuda.manual_seed_all(551)
np.random.seed(551)


def main():
    transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        # 归一化,直接除以255
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0, saturation=0, hue=0),
        transforms.RandomRotation(degrees=15),
        # transforms.Grayscale(),
        transforms.ToTensor(),
        # 减去imagenet均值 除以标准差  ,第一个epoch加上94,不加0.89
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        # Normalize(),  # 按照每张图标准化,准确率有略微提升
                                    ])

    data_image = {x: datasets.ImageFolder(root=os.path.join(IMAGE_FOLDER_PATH, x),
                                          transform=transform)
                  for x in ["train", "val"]}

    data_loader_image = {x: torch.utils.data.DataLoader(dataset=data_image[x],
                                                        batch_size=BATCH_SIZE,
                                                        shuffle=True)
                         for x in ["train", "val"]}
    classes_index = data_image["train"].class_to_idx
    class_num = len(classes_index)
    print(classes_index)
    print("训练集个数:", len(data_image["train"]))
    print("验证集个数:", len(data_image["val"]))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 加载迁移学习模型
    model = MobileNetV2(n_class=1000)
    if not CONTINUE_TRAIN:
        # add map_location='cpu' if no gpu
        model.load_state_dict(torch.load('mobilenet_v2.pth'))
        model.classifier = torch.nn.Sequential(
            torch.nn.Linear(model.last_channel, 4096),
            # torch.nn.ReLU(),
            # torch.nn.Dropout(p=0.5),
            # torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(4096, class_num)
        )
    else:
        model.classifier = torch.nn.Sequential(
            torch.nn.Linear(model.last_channel, 4096),
            # torch.nn.ReLU(),
            # torch.nn.Dropout(p=0.5),
            # torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(4096, class_num)
        )
        model.load_state_dict(torch.load('model_mobilenetv2_finetune.pth'))
    # 冻结卷积层
    for parma in model.parameters():
        parma.requires_grad = False
    for parma in model.classifier.parameters():
        parma.requires_grad = True
    model.to(device)
    cost = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters())
    n_epochs = EPOCHS
    for epoch in range(n_epochs):
        since = time.time()
        print("Epoch{}/{}".format(epoch, n_epochs))
        print("-" * 10)
        for param in ["train", "val"]:
            if param == "train":
                model.train = True
            else:
                model.train = False

            running_loss = 0.0
            running_correct = 0
            batch = 0
            for data in data_loader_image[param]:
                batch += 1
                X, y = data
                X, y = Variable(X.to(device)), Variable(y.to(device))
                # X, y = Variable(X.cuda()), Variable(y.cuda())
                optimizer.zero_grad()
                y_pred = model(X)
                _, pred = torch.max(y_pred.data, 1)

                loss = cost(y_pred, y)
                if param == "train":
                    loss.backward()
                    optimizer.step()
                running_loss += loss.item()
                running_correct += torch.sum(pred == y.data)
                if batch % 500 == 0 and param == "train":
                    print("Batch {}, Train Loss:{:.4f}, Train ACC:{:.4f}".format(
                        batch, running_loss / (4 * batch), 100 * running_correct / (4 * batch)))

            epoch_loss = running_loss / len(data_image[param])
            epoch_correct = 100 * running_correct / len(data_image[param])

            print("{}  Loss:{:.4f},  Correct{:.4f}".format(param, epoch_loss, epoch_correct))
            torch.save(model.state_dict(), "model_mobilenetv2_finetune.pth")

        now_time = time.time() - since
        print("Training time is:{:.0f}m {:.0f}s".format(now_time // 60, now_time % 60))


if __name__ == "__main__":
    # 训练集所在文件夹
    IMAGE_FOLDER_PATH = r'images'
    EPOCHS = 10
    BATCH_SIZE = 4
    CONTINUE_TRAIN = True
    main()


