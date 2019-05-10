# -*- coding: utf-8 -*-
"""
Author : Jason
Github : https://github.com/yuquant
Description : 
"""


import torch
import torchvision
from torchvision import datasets,transforms, models
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import time


def main(path):
    transform = transforms.Compose([transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    data_image = {x: datasets.ImageFolder(root=os.path.join(path, x),
                                          transform=transform)
                  for x in ["train", "val"]}

    data_loader_image = {x: torch.utils.data.DataLoader(dataset=data_image[x],
                                                        batch_size=BATCH_SIZE,
                                                        shuffle=True)
                         for x in ["train", "val"]}
    classes_index = data_image["train"].class_to_idx
    print(classes_index)
    print("训练集个数:", len(data_image["train"]))
    print("验证集个数:", len(data_image["val"]))
    model = models.vgg16(pretrained=True)
    for parma in model.parameters():
        parma.requires_grad = False
    use_gpu = torch.cuda.is_available()
    model.classifier = torch.nn.Sequential(torch.nn.Linear(25088, 4096),
                                           torch.nn.ReLU(),
                                           torch.nn.Dropout(p=0.5),
                                           torch.nn.Linear(4096, 4096),
                                           torch.nn.ReLU(),
                                           torch.nn.Dropout(p=0.5),
                                           torch.nn.Linear(4096, 2))

    if use_gpu:
        model = model.cuda()

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
                if use_gpu:
                    X, y = Variable(X.cuda()), Variable(y.cuda())
                else:
                    X, y = Variable(X), Variable(y)

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
        now_time = time.time() - since
        print("Training time is:{:.0f}m {:.0f}s".format(now_time // 60, now_time % 60))

        torch.save(model.state_dict(), "model_vgg16_finetune.pth")


if __name__ == "__main__":
    IMAGE_FOLDER_PATH = r'D:\data\dogcat'
    EPOCHS = 5
    BATCH_SIZE = 4
    main(IMAGE_FOLDER_PATH)
