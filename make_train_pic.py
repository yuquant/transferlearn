# -*- coding: utf-8 -*-
"""
Author : Jason
Github : https://github.com/yuquant
Description :  生成训练集图像数据
"""
import os
import cv2
import time


def main(class_name):
    camera = cv2.VideoCapture(0)
    camera.set(3, 224)  # 设置分辨率
    camera.set(4, 224)
    # 等待两秒
    time.sleep(2)
    i = 1
    while True:
        # 读取帧
        (ret, frame) = camera.read()
        cv2.imshow('Frame', frame)
        i += 1
        folder_path = os.path.join('images_all', class_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        # frame = cv2.resize(frame, (224,224))
        cv2.imwrite(os.path.join(folder_path, '{class_name}{num}.jpg'.format(class_name=class_name, num=i)), frame)
        time.sleep(0.1)
        # 键盘检测，检测到esc键退出
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
        # 摄像头释放
    camera.release()
    # 销毁所有窗口
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # 采集的图像类别,将文件存放到工作目录的'images_all'下
    CLASS_NAME = 'empty'
    main(CLASS_NAME)
