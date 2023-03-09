#-*-coding:utf-8-*-

import cv2
import math
import gxipy as gx
import numpy as np


def camera_set(cam,flag=True,width=1280,height=1024,exposure_value=10000.0):
    if flag:
        cam.Width.set(width)
        cam.Height.set(height)
        cam.ExposureTime.set(exposure_value)  # 曝光度设置
    else:cam.import_config_file("config_file.txt")

def rgb_improve(): # in_cam, in_rgb_image
    if cam.GammaParam.is_readable():
        gamma_value = cam.GammaParam.get()
        gamma_lut = gx.Utility.get_gamma_lut(gamma_value)  # Gamma 值:整型或浮点型,范围[0.1, 10.0],缺省值为 1
    else:
        gamma_lut = None
    if cam.ContrastParam.is_readable():
        contrast_value = cam.ContrastParam.get()
        contrast_lut = gx.Utility.get_contrast_lut(contrast_value)  # 对比度调节 Contrast 值:整型,范围[-50, 100],缺省值为 0
    else:
        contrast_lut = None
    color_correction_param = cam.ColorCorrectionParam.get()  # 颜色校正
    # contrast_lut = None
    rgb_image.image_improvement(color_correction_param, contrast_lut, gamma_lut)

def adjustRec(rec):
    width = rec[1][0]
    height = rec[1][1]
    angle = rec[2]
    while angle >= 90.0:angle -= 180.0
    while angle <- 90.0:angle += 180.0
    if angle >= 45.0:
        t = width
        width = height
        height = t
        angle -= 90.0
    elif angle < -45.0:
        t = width
        width = height
        height = t
        angle += 90.0
    return (rec[0],(width,height),angle)

def distance(light1,light2):
    return math.sqrt((light1[0][0]-light2[0][0])**2 + (light1[0][1] - light2[0][1])**2)


if __name__=="__main__":
    enemy_color = 'b'
    flag = True  # True:自定义相机参数; False:使用config_file.txt配置相机参数
    if flag:
        width,height = (1280,1024)  # 窗口大小 1024,800
        exposure_value = 5000.0  # 曝光度
    brightness_threshold = 210  # 取决于曝光度

    # 约束参数
    light_min_area = 10
    light_max_ratio = 1.0
    light_contour_min_solidity = 0.5
    light_color_detect_extend_ratio = 1.1
    light_max_angle_diff = 5.0  # 左右灯条角度差
    light_max_height_diff_ratio = 0.2
    light_max_y_diff_ratio = 2.0
    light_min_x_diff_ratio = 0.5
    armor_big_armor_ratio = 3.2
    armor_small_armor_ratio = 2
    armor_min_aspect_ratio = 1.0
    armor_max_aspect_ratio = 5.0

    # 连接设备
    device = gx.DeviceManager()
    dev_num, dev_info_list = device.update_device_list()
    if dev_num == 0:
        import sys
        sys.exit(1)
    str_sn = dev_info_list[0].get("sn")
    cam = device.open_device_by_sn(str_sn)

    #相机参数设置
    if enemy_color == 'r':exposure_value -= 1000.0
    if flag:
        camera_set(cam=cam,flag=flag,width=width,height=height,exposure_value=exposure_value)
    else:
        camera_set(cam=cam,flag=flag)

    # 帧采集
    cam.stream_on()
    while True:
        try:
            raw_image = cam.data_stream[0].get_image()
            if raw_image.get_status() == gx.GxFrameStatusList.INCOMPLETE: # 判断是否为残帧
                continue
            else:
                rgb_image = raw_image.convert("RGB")
                rgb_improve()  # 图像质量提高
                numpy_image = rgb_image.get_numpy_array()
                numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)  # opencv采用的是BGR图像，将RGB转为BGR

                # 图像处理部分
                b,_,r = cv2.split(numpy_image)
                if enemy_color == 'r':
                    grayimg = cv2.subtract(r,b)
                else:
                    grayimg = cv2.subtract(b,r)

                binimg = cv2.threshold(grayimg,brightness_threshold,255,cv2.THRESH_BINARY)[1]
                element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
                binimg = cv2.dilate(binimg,element,iterations=2)
                binimg = cv2.GaussianBlur(binimg, (5, 5), 0)  # 高斯滤波

                lightContours = cv2.findContours(binimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
                # cv2.drawContours(numpy_image,lightContours,-1,(0,255,0),1)

                # 灯条筛选
                lightInfos = []
                for cnt in lightContours:
                    area = cv2.contourArea(cnt)
                    if len(cnt) <= 5 or area < light_min_area or area > 10000:continue
                    ellipse = cv2.fitEllipse(cnt)  # [(x,y), (a,b), angle] a,b短长轴,灯条宽高
                    ellipse = adjustRec(ellipse)
                    ellipse_area = math.pi * ellipse[1][0] * ellipse[1][1] / 4  # 拟合椭圆面积
                    if ellipse[1][0] / ellipse[1][1] > light_max_ratio or area / ellipse_area < light_contour_min_solidity:continue
                    lightInfos.append([ellipse[0], (ellipse[1][0] * light_color_detect_extend_ratio, ellipse[1][1] * light_color_detect_extend_ratio), ellipse[2]])

                light_size = len(lightInfos)
                if light_size > 0:
                    lightInfos = sorted(lightInfos,key=lambda x:x[0][0]) # 按照x从小到大排序
                    for i in range(light_size):
                        for j in range(i+1,light_size):
                            leftLight = lightInfos[i]
                            rightLight = lightInfos[j]

                            # 角差
                            angleDiff = abs(leftLight[2] - rightLight[2])
                            # 长度差比率
                            LenDiff_ratio = abs(leftLight[1][1] - rightLight[1][1]) / max(leftLight[1][1],rightLight[1][1])
                            # 筛选
                            if angleDiff > light_max_angle_diff or LenDiff_ratio > light_max_height_diff_ratio:continue

                            # 左右灯条相距距离
                            dis = distance(leftLight,rightLight)
                            # 左右灯条长度灯平均值
                            meanLen = (leftLight[1][1] + rightLight[1][1]) / 2
                            # 左右灯条中心点y的差值
                            yDiff = abs(leftLight[0][1] - rightLight[0][1])
                            # y差比率
                            yDiff_ratio = yDiff / meanLen
                            # 左右灯条中心点x的差值
                            xDiff = abs(leftLight[0][0] - rightLight[0][0])
                            # x差比率
                            xDiff_ratio = xDiff / meanLen
                            # 相距距离与灯条长度比值
                            ratio = dis / meanLen

                            if yDiff_ratio > light_max_y_diff_ratio or xDiff_ratio < light_min_x_diff_ratio or ratio > armor_max_aspect_ratio or ratio < armor_min_aspect_ratio:continue
                            roi_x = int((leftLight[0][0] + rightLight[0][0]) / 2)
                            roi_y = int((leftLight[0][1] + rightLight[0][1]) / 2)
                            pos1 = (int(roi_x-dis/2), int(roi_y-meanLen))
                            pos2 = (int(roi_x+dis/2), int(roi_y+meanLen))
                            # cv2.rectangle(numpy_image, pos1, pos2, (0, 255, 0), 2)
                            # cv2.drawMarker(numpy_image, (roi_x, roi_y), (0, 255, 0), cv2.MARKER_CROSS, thickness=3)
                            binimg = numpy_image[pos1[1]:pos2[1],pos1[0]:pos2[0]]

                            # cv2.ellipse(numpy_image, leftLight, (0, 255, 0), 1)
                            # cv2.ellipse(numpy_image, rightLight, (0, 255, 0), 1)



                    # cv2.ellipse(numpy_image,ellipse,(0,255,0),1)
                    # x, y, w, h = cv2.boundingRect(cnt)
                    # if h / w >= 2 and h / w <= 5 and area <= 3000 and area > 500  and w * h <= 5000:
                    #     cv2.rectangle(numpy_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    #     print(area)
                    # # print(x,y,w,h)
                    # if w / h < 1:
                    #     continue
                    # if h / w > 0.4:
                    #     continue


                # 显示图像
                cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
                cv2.imshow('frame', numpy_image)
                cv2.namedWindow('grayimg', cv2.WINDOW_NORMAL)
                cv2.imshow('grayimg',binimg)
                if cv2.waitKey(1) & 0xFF == 27:break
        except:continue

    cv2.destroyAllWindows()
    cam.stream_off()
    cam.close_device()
