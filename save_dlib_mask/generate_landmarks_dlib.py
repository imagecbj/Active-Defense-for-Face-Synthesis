import os
from collections import OrderedDict
import dlib
import numpy as np

# import torch
import json
import cv2


# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 初始化dlib的人脸检测器和形状预测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./dlib_model/shape_predictor_68_face_landmarks.dat')

# 定义保存人脸标记的函数
def save_landmarks(input_dir, save_dir):
    # 初始化存储图像名称和人脸标记的有序字典
    all_landmarks = OrderedDict()
    all_images = OrderedDict()

    # 列出输入目录下的所有文件名
    image_names = os.listdir(input_dir)
    for name in image_names:
        # 跳过非png文件
        if name.find('png') == -1:
            continue
        # 读取图像，将BGR格式转换为RGB格式，并调整大小至256x256
        image = cv2.cvtColor(cv2.imread(os.path.join(input_dir, name), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256))
        # 存储处理后的图像
        all_images[name] = image

    # 从有序字典中提取图像名称和图像数据
    all_image_names = list(all_images.keys())
    all_images = list(all_images.values())
    # 遍历所有图像，提取人脸标记
    for i in range(0, len(all_images)):
        try:
            # 使用dlib检测人脸并预测人脸标记
            rect = detector(all_images[i])[0]
            sp = predictor(all_images[i], rect)
            landmarks = np.array([[p.x, p.y] for p in sp.parts()])
            # 存储人脸标记
            all_landmarks[all_image_names[i]] = landmarks
        except Exception as e:
            # 如果出现错误，打印错误信息
            print(e)
            pass

    # 构建输出文件的名称并打开文件进行写入
    json_name = input_dir.split('/')[-1] + '.json'
    with open(os.path.join(save_dir, json_name), "w", encoding='utf-8') as out_file:
        # 将每个图像的名称和对应的人脸标记写入文件
        for index, name in enumerate(all_landmarks.keys()):
            dict = {"image_name": name, "landmarks": all_landmarks[name].astype(np.int16).tolist()}
            dict = json.dumps(dict) + "\n"
            out_file.write(dict)



