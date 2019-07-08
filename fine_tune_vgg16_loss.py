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


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    @staticmethod
    def normalize(tensor):
        """Normalize a tensor image with mean and standard deviation.

        See ``Normalize`` for more details.

        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
            mean (sequence): Sequence of means for each channel.
            std (sequence): Sequence of standard deviations for each channely.

        Returns:
            Tensor: Normalized Tensor image.
        """
        # TODO: make efficient

        for t in tensor:
            m = t.mean()
            s = t.std()
            t.sub_(m).div_(s)
        return tensor

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        tensor = self.normalize(tensor)
        return tensor


def main():
    transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        # 归一化,直接除以255
        transforms.RandomHorizontalFlip(),  # 随机水平反转
        transforms.RandomVerticalFlip(),  # 随机垂直反转
        transforms.ColorJitter(brightness=0.1, contrast=0, saturation=0, hue=0),  # 亮度随机浮动10%
        transforms.RandomRotation(degrees=15),  # 随机旋转正负15度
        # transforms.Grayscale(num_output_channels=3),  # 彩色图变成灰度图
        transforms.ToTensor(),
        # 减去imagenet均值 除以标准差  ,第一个epoch加上94,不加0.89
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        Normalize(),  # 按照每张图标准化,准确率有略微提升
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
    if not CONTINUE_TRAIN:
        model = models.vgg16(pretrained=True)
        model.classifier = torch.nn.Sequential(
            torch.nn.Linear(25088, 4096),
            # torch.nn.ReLU(),
            # torch.nn.Dropout(p=0.5),
            # torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(4096, class_num)
        )
    else:
        model = models.vgg16(pretrained=False)
        model.classifier = torch.nn.Sequential(
            torch.nn.Linear(25088, 4096),
            # torch.nn.ReLU(),
            # torch.nn.Dropout(p=0.5),
            # torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(4096, class_num)
        )
        state_dict = torch.load(WEIGHT_NAME)  # add map_location='cpu' if no gpu
        model.load_state_dict(state_dict)
    for parma in model.parameters():
        parma.requires_grad = False
    for parma in model.classifier.parameters():
        parma.requires_grad = True
    model.to(device)
    # cost = torch.nn.CrossEntropyLoss()
    cost = torch.nn.MultiLabelSoftMarginLoss()  # 损失函数没什么大问题
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
            multi_correct = 0
            for data in data_loader_image[param]:
                batch += 1
                X, y = data
                y_onehot = y.clone()
                # y_onehot = y
                y_onehot = y_onehot.numpy()
                y_onehot = (np.arange(3) == y_onehot[:, None]).astype(np.float32)
                y_onehot = torch.from_numpy(y_onehot)

                X, y_onehot,y = Variable(X.to(device)), Variable(y_onehot.to(device)), Variable(y.to(device))
                # X, y = Variable(X.cuda()), Variable(y.cuda())
                optimizer.zero_grad()
                y_pred = model(X)
                # y_pred = torch.sigmoid(y_pred)  # 加这个分类不容易收敛
                _, pred = torch.max(y_pred.data, 1)
                multi_pred = y_pred.data.clone()
                multi_pred[multi_pred > 1] = 1
                multi_pred[multi_pred < 0] = 0
                loss = cost(y_pred, y_onehot)
                if param == "train":
                    loss.backward()
                    optimizer.step()
                running_loss += loss.item()
                running_correct += torch.sum(pred == y.data)
                multi_correct += torch.sum(multi_pred == y_onehot)
                if batch % 500 == 0 and param == "train":
                    print("Batch {}, Train Loss:{:.4f}, Train ACC:{:.4f}".format(
                        batch, running_loss / (4 * batch), 100 * running_correct / (4 * batch)))

            epoch_loss = running_loss / len(data_image[param])
            epoch_correct = 100 * running_correct / len(data_image[param])
            epoch_correct_multi = 100 * multi_correct / len(data_image[param]) / class_num
            print("{}  Loss:{:.4f},  Correct{:.4f}, Multi Correct{:.4f}".format(param, epoch_loss, epoch_correct, epoch_correct_multi))
            torch.save(model.state_dict(), WEIGHT_NAME)

        now_time = time.time() - since
        print("Training time is:{:.0f}m {:.0f}s".format(now_time // 60, now_time % 60))


if __name__ == "__main__":
    # 训练集所在文件夹
    import copy
    IMAGE_FOLDER_PATH = r'images'
    EPOCHS = 10
    BATCH_SIZE = 4
    CONTINUE_TRAIN = True
    WEIGHT_NAME = "model_vgg16_finetune_sigmoid.pth"
    main()
