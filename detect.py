# coding: utf-8
"""
分类器检查
"""
import numpy as np
import cv2
import torch
import torchvision
from MobileNetV2 import MobileNetV2


def load_model_mobile_net():
    model = MobileNetV2(n_class=1000)
    model.classifier = torch.nn.Sequential(
        torch.nn.Linear(model.last_channel, 4096),
        # torch.nn.ReLU(),
        # torch.nn.Dropout(p=0.5),
        # torch.nn.Linear(4096, 4096),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=0.5),
        torch.nn.Linear(4096, len(CLASS_DICT))
    )
    model.load_state_dict(torch.load('model_mobilenetv2_finetune_best.pth'))
    return model


def load_model_vgg16():
    model = torchvision.models.vgg16(pretrained=False)
    model.classifier = torch.nn.Sequential(torch.nn.Linear(25088, 4096),
                                           # torch.nn.ReLU(),
                                           # torch.nn.Dropout(p=0.5),
                                           # torch.nn.Linear(4096, 4096),
                                           torch.nn.ReLU(),
                                           torch.nn.Dropout(p=0.5),
                                           torch.nn.Linear(4096, len(CLASS_DICT)))
    model.load_state_dict(torch.load('model_vgg16_finetune_best.pth'))

    return model


def transform(arr):
    arr = cv2.resize(arr, dsize=(224, 224))
    arr = (arr - arr.mean()) / arr.std()
    arr = np.transpose(arr, axes=(2, 0, 1))
    arr = arr[np.newaxis].astype(np.float32)
    # arr = torch.Tensor(arr)
    arr = torch.from_numpy(arr)
    arr = torch.torch.autograd.Variable(arr)
    return arr


def main():
    camera = cv2.VideoCapture(0)
    camera.set(3, 224)  # 设置分辨率
    camera.set(4, 224)
    # 等待两秒
    # 加载模型
    if MODEL_NAME == 'VGG16':
        model = load_model_vgg16()
    elif MODEL_NAME == 'MOBILE_NET':
        model = load_model_mobile_net()
    model.eval()
    # 遍历每一帧
    target_last = ''
    while True:
        # 读取帧
        (ret, frame) = camera.read()
        # 判断是否成功打开摄像头
        if not ret:
            print('No Camera')
            continue
        # frame = cv2.imread('jdb.jpg')
        # 按照训练时所作的操作处理图像
        inputs = transform(frame)
        # 将图像输入到模型中预测
        outputs = model(inputs)
        _, pred = torch.max(outputs, 1)
        target = CLASS_DICT[pred.item()]
        # if target_last == target:
        #     print(target)
        #     cv2.putText(frame, target, (20, 20), cv2.FONT_HERSHEY_COMPLEX, 1,
        #                 (0, 255, 0), 1)  # print(yanhe)
        target_last = target
        print(target)
        # 写入文字到图像中
        cv2.putText(frame, target, (20, 20), cv2.FONT_HERSHEY_COMPLEX, 1,
                    (0, 255, 0), 1)
        cv2.imshow('Frame', frame)
        # 键盘检测，检测到esc键退出
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
    # 摄像头释放
    camera.release()
    # 销毁所有窗口
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # 类别输出的数值对应的类别的字典
    CLASS_DICT = {0: 'circle', 1: 'nothing', 2: 'rectangle'}
    # 选择分类模型
    MODEL_NAME = 'VGG16'  # 'VGG16' 或者 'MOBILE_NET'
    main()
