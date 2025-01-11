
import cv2
import math
import json
import random
import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
from PIL import Image
from collections import OrderedDict
from DeepFakeMask import dfl_full,facehull,components,extended

import os


    

class ff_df_Dataloader(Dataset):
    def __init__(self, root_path="", fake_video_names=[], source_video_names=[], phase='train', test_frame_nums=3,
                 transform=None, size=(256, 256)):
        # 确保传入的阶段参数是预定义的选项之一
        assert phase in ['train', 'valid', 'test']
        self.root_path = root_path  # 设置根路径
        self.fake_video_names = []  # 初始化伪造视频名称列表
        self.source_video_names = []  # 初始化源视频名称列表
        self.source_video_landmarks = OrderedDict()  # 初始化源视频的人脸标记字典
        # 遍历伪造视频和源视频的名称列表
        for fake_video_name, source_video_name in tqdm(zip(fake_video_names, source_video_names)):
            aa = source_video_name.split('/')  # 分割源视频路径
            # 构建与源视频对应的JSON文件路径

            if len(aa) == 4 :
                json_file = os.path.join('Face', aa[0], aa[1], aa[2],  'dlib_landmarks' ,aa[3] + '.json')
            else:
                json_file = os.path.join('Face', aa[0],  'dlib_landmarks' ,aa[1] + '.json')
            json_path = os.path.join(self.root_path, json_file)  # 拼接完整的JSON文件路径
            # 如果JSON文件存在
            if os.path.isfile(json_path):
                all_landmarks = self.load_landmarks(json_path)  # 加载人脸标记
                # 如果加载到的人脸标记不为空
                if len(all_landmarks) != 0:
                    self.source_video_landmarks[source_video_name] = all_landmarks  # 保存人脸标记
                    self.fake_video_names.append(fake_video_name)  # 添加伪造视频名称到列表
                    self.source_video_names.append(source_video_name)  # 添加源视频名称到列表
        self.phase = phase  # 设置阶段
        self.test_frame_nums = test_frame_nums  # 设置测试帧数
        self.transform = transform  # 设置转换方法
        self.size = size  # 设置图像尺寸
        # 如果阶段不是训练，加载图像名称
        if phase != 'train':
            self.fake_image_names, self.source_image_names = self.load_image_name()
        else:
            # 如果是训练阶段，打印测试视频的数量
            # self.fake_image_names, self.source_image_names = self.load_image_name()
            print('The number of test videos is : %d' % len(fake_video_names))


    def load_image_name(self):
        fake_image_names = []  # 初始化伪造图像名称列表
        source_image_names = []  # 初始化源图像名称列表
        # 遍历伪造视频名称列表
        for idx, fake_video_name in tqdm(enumerate(self.fake_video_names)):
            random.seed(2021)  # 设置随机种子以确保结果可重复
            video_path = os.path.join(self.root_path,'Face', fake_video_name)  # 构建伪造视频的完整路径
            source_video_name = self.source_video_names[idx]  # 获取对应的源视频名称
            all_frame_names = os.listdir(video_path)  # 列出视频路径下的所有帧名称
            frame_names = []  # 初始化选定的帧名称列表
            # 遍历所有帧名称
            for image_name in all_frame_names:
                # 选择每十帧的一帧，并确保该帧在源视频的人脸标记中存在
                if int(image_name.split('/')[-1].replace('.png', '')) % 10 == 0 and \
                        self.source_video_landmarks[source_video_name].get(image_name) is not None:
                    frame_names.append(image_name)  # 将符合条件的帧名称添加到列表
            # 如果选定的帧数量大于测试帧数，则从中随机抽取测试帧数数量的帧
            if len(frame_names) > self.test_frame_nums:
                # frame_names = frame_names[:self.test_frame_nums]
                frame_names = random.sample(frame_names, self.test_frame_nums)
            # 将选定的帧名称添加到伪造图像和源图像名称列表中
            for image_name in frame_names:
                fake_image_names.append(os.path.join(fake_video_name, image_name))
                source_image_names.append(os.path.join(source_video_name, image_name))
        return fake_image_names, source_image_names  # 返回伪造图像和源图像名称列表

    def load_landmarks(self, landmarks_file):
        # 初始化有序字典来存储所有的人脸标记数据
        all_landmarks = OrderedDict()
        # 打开传入的人脸标记JSON文件
        with open(landmarks_file, "r", encoding="utf-8") as file:
            line = file.readline()  # 读取文件的一行
            while line:
                line = json.loads(line)  # 将读取的行转换为JSON格式
                # 将图像名称和对应的标记数据存储到字典中
                all_landmarks[line["image_name"]] = np.array(line["landmarks"])
                line = file.readline()  # 继续读取下一行
        return all_landmarks  # 返回存储了所有人脸标记的字典

    def get_label(self, path):
        # 根据路径中的关键字判断标签
        if path.find('ORIGINAL') != -1:
            label = 0  # 真实视频
        elif path.find('Deepfakes') != -1:
            label = 1  # Deepfakes视频
        elif path.find('FaceSwap') != -1:
            label = 2  # FaceSwap视频
        elif path.find('FaceShifter') != -1:
            label = 3  # FaceShifter视频
        elif path.find('Face2Face') != -1:
            label = 4  # Face2Face视频
        elif path.find('NeuralTextures') != -1:
            label = 5  # NeuralTextures视频
        return label  # 返回视频的标签

    def read_png(self, image_path):
        # 使用OpenCV读取图像文件
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        # 将BGR格式的图像转换为RGB格式
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image  # 返回转换后的图像

    def __getitem__(self, index):
        # 如果处于训练阶段
        if self.phase == 'train':
            # 获取对应索引的伪造视频和源视频名称
            fake_video_name = self.fake_video_names[index]
            source_video_name = self.source_video_names[index]
            # 获取源视频的所有人脸标记
            all_landmarks = self.source_video_landmarks[source_video_name]

            # 从所有人脸标记中随机选择一个图像名称
            image_name = random.sample(list(all_landmarks.keys()), 1)[0]
            # image_name = list(all_landmarks.keys())[0]
            # 构建伪造图像和源图像的路径
            fake_video_path = os.path.join(self.root_path,'Face', fake_video_name)
            fake_image_path = os.path.join(fake_video_path, image_name)
            # 读取伪造图像
            fake_image = self.read_png(fake_image_path)

            source_image_path = os.path.join(self.root_path,'Face', source_video_name, image_name)
            # 读取源图像
            source_image = self.read_png(source_image_path)

            # 生成人脸遮罩
            face_mask = facehull(landmarks=all_landmarks[image_name].astype('int32'),
                                 face=cv2.resize(source_image, self.size), channels=3).mask
            # import matplotlib.pyplot as plt
            #
            # wwww = face_mask
            #
            #
            # plt.imshow(wwww)  # 使用灰度色彩映射来更好地展示二值图像
            # plt.colorbar()  # 显示色标，可选
            # plt.show()

            # 应用转换到伪造图像和源图像
            fake_image = self.transform(image=fake_image)["image"]
            #face_data = self.transform(image=source_image, mask=face_mask)
            source_image = self.transform(image=source_image)["image"]
            face_mask = self.transform(image=face_mask)["image"]

            return fake_image, source_image, face_mask

        # 如果处于验证或测试阶段
        else:
            # 构建伪造图像和源图像的路径
            fake_image_path = os.path.join(self.root_path,'Face', self.fake_image_names[index])
            source_image_path = os.path.join(self.root_path,'Face', self.source_image_names[index])
            source_video_name = '/'.join(source_image_path.split('/')[9:-1])
            all_landmarks = self.source_video_landmarks[source_video_name]

            # 读取伪造图像和源图像
            fake_image = self.read_png(fake_image_path)
            source_image = self.read_png(source_image_path)
            # 生成人脸遮罩
            face_mask = facehull(landmarks=all_landmarks[source_image_path.split('/')[-1]].astype('int32'),
                                 face=cv2.resize(source_image, self.size), channels=3).mask

            # 应用转换到伪造图像和源图像
            fake_image = self.transform(image=fake_image)["image"]
            # face_data = self.transform(image=source_image, mask=face_mask)
            source_image = self.transform(image=source_image)["image"]
            face_mask = self.transform(image=face_mask)["image"]

            return fake_image, source_image, face_mask

    def __len__(self):
        # 如果处于训练阶段，返回伪造视频名称列表的长度
        if self.phase == 'train':
            return len(self.source_video_names)
        # 如果处于验证或测试阶段，返回伪造图像名称列表的长度
        else:
            return len(self.fake_image_names)

