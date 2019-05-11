# -*- coding: utf-8 -*-
"""
Author : Jason
Github : https://github.com/yuquant
Description : 
"""


import torch
import torchvision
from torchvision import datasets, transforms, models
import os
import copy
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import time
torch.manual_seed(551)
torch.cuda.manual_seed_all(551)
np.random.seed(551)


def train_model(model, criterion, optimizer, data_image, dataloaders, device, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    history_loss = {
        "train": [],
        "val": []
    }
    history_accuracy = {
        "train": [],
        "val": []
    }

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                # scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.long()
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                # print(running_corrects)

            epoch_loss = running_loss / len(data_image[phase])
            epoch_acc = running_corrects.double() / len(data_image[phase])

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # History Statistics
            history_loss[phase].append(epoch_loss)
            history_accuracy[phase].append(epoch_acc.item())

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, history_loss, history_accuracy


def main(path):
    transform = transforms.Compose([transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                    ])

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
    model = models.resnet50(pretrained=True)
    for parma in model.parameters():
        parma.requires_grad = False
    use_gpu = torch.cuda.is_available()
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Sequential(torch.nn.Linear(num_ftrs, 4096),
                                           torch.nn.ReLU(),
                                           torch.nn.Dropout(p=0.5),
                                           torch.nn.Linear(4096, 4096),
                                           torch.nn.ReLU(),
                                           torch.nn.Dropout(p=0.5),
                                           torch.nn.Linear(4096, 2))

    if use_gpu:
        model = model.cuda()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters())
    model, history_loss, history_accuracy = train_model(model=model,
                                                        criterion=criterion,
                                                        optimizer=optimizer,
                                                        data_image=data_image,
                                                        dataloaders=data_loader_image,
                                                        device=DEVICE,
                                                        num_epochs=EPOCHS)

    torch.save(model.state_dict(), "model_resnet50_finetune.pth")


if __name__ == "__main__":
    IMAGE_FOLDER_PATH = r'D:\projects\detect\images'
    EPOCHS = 5
    BATCH_SIZE = 4
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    main(IMAGE_FOLDER_PATH)
