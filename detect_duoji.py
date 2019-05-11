# -*- coding: utf-8 -*-
"""
Author : Jason
Github : https://github.com/yuquant
Description :   舵机控制代码参考
"""
from collections import deque
import numpy as np
import cv2
import time
import torch
import torchvision
import RPi.GPIO as GPIO
import time
import signal
import atexit


def load_model():
    model = torchvision.models.vgg16(pretrained=False)
    model.classifier = torch.nn.Sequential(torch.nn.Linear(25088, 4096),
                                           # torch.nn.ReLU(),
                                           # torch.nn.Dropout(p=0.5),
                                           # torch.nn.Linear(4096, 4096),
                                           torch.nn.ReLU(),
                                           torch.nn.Dropout(p=0.5),
                                           torch.nn.Linear(4096, len(CLASS_DICT)))
    model.load_state_dict(torch.load('model_vgg16_finetune.pth'))

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
    # 舵机初始化
    atexit.register(GPIO.cleanup)
    servopin = 21
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(servopin, GPIO.OUT, initial=False)
    p = GPIO.PWM(servopin, 50)  # 50HZ    
    p.start(0)
    # 开启相机
    camera = cv2.VideoCapture(0)
    camera.set(3, 224)  # 设置分辨率
    camera.set(4, 224)
    # 等待两秒
    # 加载模型
    model = load_model()
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
        inputs = transform(frame)
        outputs = model(inputs)
        _, pred = torch.max(outputs, 1)
        target = CLASS_DICT[pred.item()]
        print(target)
        cv2.putText(frame, target, (20, 20), cv2.FONT_HERSHEY_COMPLEX, 1,
                    (0, 255, 0), 1)
        cv2.imshow('Frame', frame)
        if target_last != target:
            if pred.item() == 0:
                # 舵机旋转180度,再恢复原状
                for i in range(0, 181, 10):
                    p.ChangeDutyCycle(2.5 + 10 * i / 180)  # 设置转动角度
                    time.sleep(0.02)  # 等该20ms周期结束
                    p.ChangeDutyCycle(0)  # 归零信号
                    time.sleep(0.2)

                for i in range(181, 0, -10):
                    p.ChangeDutyCycle(2.5 + 10 * i / 180)
                    time.sleep(0.02)
                    p.ChangeDutyCycle(0)
                    time.sleep(0.2)
            if pred.item() == 2:
                # 舵机反向旋转180度,再恢复原状
                for i in range(181, 0, -10):
                    p.ChangeDutyCycle(2.5 + 10 * i / 180)
                    time.sleep(0.02)
                    p.ChangeDutyCycle(0)
                    time.sleep(0.2)
                for i in range(0, 181, 10):
                    p.ChangeDutyCycle(2.5 + 10 * i / 180)  # 设置转动角度
                    time.sleep(0.02)  # 等该20ms周期结束
                    p.ChangeDutyCycle(0)  # 归零信号
                    time.sleep(0.2)
        target_last = target

        # 键盘检测，检测到esc键退出
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
    # 摄像头释放
    camera.release()
    # 销毁所有窗口
    cv2.destroyAllWindows()


if __name__ == "__main__":
    CLASS_DICT = {0: 'circle', 1: 'empty', 2: 'rectangle'}
    main()
