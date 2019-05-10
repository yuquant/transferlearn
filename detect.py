# coding: utf-8
from collections import deque
import numpy as np
import cv2
import time

if __name__ == "__main__":
    # 设定黄色阈值，HSV空间
    yellowLower = np.array([78, 43, 46])
    yellowUpper = np.array([99, 255, 255])

    redLower = np.array([156, 43, 46])
    redUpper = np.array([180, 255, 255])


    def makeColor(color1, color2, txt):
        global ret, frame
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # 根据阈值构建掩膜
        mask = cv2.inRange(hsv, color1, color2)
        # 腐蚀操作
        mask = cv2.erode(mask, None, iterations=2)
        # 膨胀操作，其实先腐蚀再膨胀的效果是开运算，去除噪点
        mask = cv2.dilate(mask, None, iterations=2)
        # 轮廓检测
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        # 初始化瓶盖圆形轮廓质心
        center = None
        # 如果存在轮廓
        if len(cnts) > 0:
            # 找到面积最大的轮廓
            c = max(cnts, key=cv2.contourArea)
            # 确定面积最大的轮廓的外接圆
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            # 计算轮廓的矩
            M = cv2.moments(c)
            # 计算质心
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            # 只有当半径大于10时，才执行画图
            if radius > 30:
                cv2.polylines(frame, c, True, (0, 0, 255))
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
                cv2.putText(frame, txt, (int(x) + int(radius), int(y) + int(radius)), cv2.FONT_HERSHEY_COMPLEX, 1,
                            (0, 255, 0), 1)
                # 把质心添加到pts中，并且是添加到列表左侧
                pts.appendleft(center)
                return ((x, y), radius)
        return -1


    mybuffer = 64
    pts = deque(maxlen=mybuffer)
    # 打开摄像头
    camera = cv2.VideoCapture(0)
    # 等待两秒
    time.sleep(2)
    # 遍历每一帧，检测
    while True:
        # 读取帧
        time.sleep(0.5)
        result = []
        (ret, frame) = camera.read()
        ret = 1
        # 判断是否成功打开摄像头
        if not ret:
            print('No Camera')
            break
        # frame = cv2.imread('jdb.jpg')
        ball = makeColor(yellowLower, yellowUpper, 'ball')
        # print(ball)
        yanhe = makeColor(redLower, redUpper, 'circle')
        # print(yanhe)
        if yanhe != -1:
            flag = '烟盒'
        elif ball != -1:
            flag = '网球'
        else:
            flag = '没有检测到物体'
        print(flag)
        cv2.imshow('Frame', frame)
        print(flag)  # 每一帧的结果
        # 键盘检测，检测到esc键退出
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
    # 摄像头释放
    camera.release()
    # 销毁所有窗口
    cv2.destroyAllWindows()
