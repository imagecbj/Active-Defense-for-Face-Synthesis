import torch
import json
import cv2
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from PIL import Image

from facenet_pytorch import MTCNN

import os


os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def extract_video(input_dir, model, scale=1.3, smallest_h=0, smallest_w=0, gp=10):
    reader = cv2.VideoCapture(input_dir)  # 打开视频文件
    frames_num = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取视频的总帧数
    batch_size = 2048  # 定义批处理大小
    face_boxes = []  # 初始化存储面部边界框的列表
    face_images = []  # 初始化存储面部图像的列表
    face_index = []  # 初始化存储面部图像在视频中的索引的列表
    original_frames = OrderedDict()  # 初始化存储原始帧的有序字典
    halve_frames = OrderedDict()  # 初始化存储缩小一半尺寸的帧的有序字典
    index_frames = OrderedDict()  # 初始化存储帧索引的有序字典
    for i in range(frames_num):  # 遍历每一帧
        reader.grab()  # 抓取下一帧
        success, frame = reader.retrieve()  # 从视频流中提取帧
        frame_shape = frame.shape  # 获取帧的尺寸
        # 调整帧的高度以匹配最小高度
        if smallest_h != frame_shape[0]:
            diff = frame_shape[0] - smallest_h
            m = diff // 2
            frame = frame[m:-m, :, :]
        # 调整帧的宽度以匹配最小宽度
        if smallest_w != frame_shape[1]:
            diff = frame_shape[1] - smallest_w
            m = diff // 2
            frame = frame[:, m:-m, :]
        # 每间隔gp帧进行处理
        if i % gp == 0:
            if not success:
                continue
            original_frames[i] = frame  # 存储原始帧
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 将帧从BGR转换为RGB
            frame = Image.fromarray(frame)  # 将数组转换为图像
            frame = frame.resize(size=[s // 2 for s in frame.size])  # 缩小帧尺寸为一半
            halve_frames[i] = frame  # 存储缩小后的帧
            index_frames[i] = i  # 存储帧索引

    original_frames = list(original_frames.values())  # 将原始帧的有序字典转换为列表
    halve_frames = list(halve_frames.values())  # 将缩小帧的有序字典转换为列表
    index_frames = list(index_frames.values())  # 将帧索引的有序字典转换为列表
    print(input_dir[-7:])  # 打印视频文件名的最后7个字符
    # 批量处理帧以检测面部
    for i in range(0, len(halve_frames), batch_size):
        batch_boxes, batch_probs, batch_points = model.detect(halve_frames[i:i + batch_size], landmarks=True)  # 使用MTCNN检测面部
        None_array = np.array([], dtype=np.int16)  # 初始化一个空的numpy数组，但未使用
        for index, bbox in enumerate(batch_boxes):
            if bbox is not None:
                pass
            else:
                print("no face in:{}_{}".format(input_dir[-7:],index))
                batch_boxes[index] = batch_boxes[index-1]  # 如果当前帧未检测到面部，则使用上一帧的信息
                batch_probs[index] = batch_probs[index-1]
                batch_points[index] = batch_points[index-1]
                continue

        batch_boxes, batch_probs, batch_points = model.select_boxes(batch_boxes, batch_probs, batch_points,
                                                                    halve_frames[i:i + batch_size],
                                                                    method="probability")  # 选择最有可能的面部边界框
        # 提取面部并保存
        for index, bbox in enumerate(batch_boxes):
            if bbox is not None:
                xmin, ymin, xmax, ymax = [int(b * 2) for b in bbox[0, :]]  # 将边界框坐标放大两倍，还原到原始尺寸
                w = xmax - xmin  # 计算宽度
                h = ymax - ymin  # 计算高度
                size_bb = int(max(w, h) * scale)  # 根据scale参数调整边界框大小
                center_x, center_y = (xmin + xmax) // 2, (ymin + ymax) // 2  # 计算边界框中心点

                # 检查边界框坐标是否超出帧边界
                xmin = max(int(center_x - size_bb // 2), 0)
                ymin = max(int(center_y - size_bb // 2), 0)
                size_bb = min(original_frames[i:i + batch_size][index].shape[1] - xmin, size_bb)
                size_bb = min(original_frames[i:i + batch_size][index].shape[0] - ymin, size_bb)

                face_index.append(index_frames[i:i + batch_size][index])  # 存储面部索引
                face_boxes.append([ymin, ymin + size_bb, xmin, xmin + size_bb])  # 存储面部边界框
                crop = original_frames[i:i + batch_size][index][ymin:ymin + size_bb, xmin:xmin + size_bb]  # 裁剪面部图像
                face_images.append(crop)  # 存储面部图像
            else:
                continue

    return face_images, face_boxes, face_index  # 返回面部图像、面部边界框和面部索引



def get_smallest_hw(video_root_path, real_sub_path, real_video, deepfake_sub_path, fake_video):
    # 拼接真实视频的完整路径
    #real_video_path = os.path.join(video_root_path, real_sub_path, real_video)
    # 读取真实视频的第一帧
    reader = cv2.VideoCapture(os.path.join(video_root_path, real_sub_path, real_video))
    reader.grab()
    success, frame = reader.retrieve()
    # 获取真实视频帧的尺寸
    frame_real_shape = frame.shape
    del reader  # 删除视频读取对象以释放资源

    # 重复上述步骤，获取深度伪造视频的帧尺寸
    reader = cv2.VideoCapture(os.path.join(video_root_path, deepfake_sub_path, fake_video))
    reader.grab()
    success, frame = reader.retrieve()
    frame_deepfakes_shape = frame.shape
    del reader

    # 对其他类型的深度伪造视频重复相同步骤
    # Face2Face
    reader = cv2.VideoCapture(os.path.join(video_root_path, deepfake_sub_path.replace('Deepfakes', 'Face2Face'), fake_video))
    reader.grab()
    success, frame = reader.retrieve()
    frame_face2face_shape = frame.shape
    del reader

    # NeuralTextures
    reader = cv2.VideoCapture(os.path.join(video_root_path, deepfake_sub_path.replace('Deepfakes', 'NeuralTextures'), fake_video))
    reader.grab()
    success, frame = reader.retrieve()
    frame_NeuralTextures_shape = frame.shape
    del reader

    # FaceSwap
    reader = cv2.VideoCapture(os.path.join(video_root_path, deepfake_sub_path.replace('Deepfakes', 'FaceSwap'), fake_video))
    reader.grab()
    success, frame = reader.retrieve()
    frame_FaceSwap_shape = frame.shape
    del reader



    # 计算所有视频中最小的高度和宽度
    smallest_h = min(frame_real_shape[0], frame_deepfakes_shape[0], frame_face2face_shape[0],
                     frame_NeuralTextures_shape[0], frame_FaceSwap_shape[0])
    smallest_w = min(frame_real_shape[1], frame_deepfakes_shape[1], frame_face2face_shape[1],
                     frame_NeuralTextures_shape[1], frame_FaceSwap_shape[1])

    # 返回计算得到的最小高度和宽度
    return smallest_h, smallest_w



def main(video_root_path, image_root_path, real_sub_path, deepfake_sub_path, real_videos, fake_videos):
    scale = 1.3  # 设置缩放比例

    # 根据CUDA可用性选择设备
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))  # 打印当前使用的设备
    # 初始化MTCNN检测器
    detector = MTCNN(margin=0, thresholds=[0.6, 0.7, 0.7], device=device)

    # 遍历真实视频列表
    for idx in tqdm(range(len(real_videos))):
        # 构造真实视频的完整路径
        real_video_path = os.path.join(video_root_path, real_sub_path, real_videos[idx])
        # 检查目标路径是否存在
        check_path = os.path.join(image_root_path, real_sub_path, real_videos[idx])

        # 如果路径已存在，则跳过当前视频
        if os.path.exists(check_path):
            print("{} is existed".format(real_videos[idx]))
            continue

        # 获取当前视频组合中最小的高度和宽度
        smallest_h, smallest_w = get_smallest_hw(video_root_path, real_sub_path, real_videos[idx], deepfake_sub_path, fake_videos[idx])

        # 从真实视频中提取面部图像
        face_images, face_boxes, face_index = extract_video(real_video_path, detector, scale=scale, smallest_h=smallest_h, smallest_w=smallest_w)

        # 准备保存提取图像的目录
        temp_save_root_path = os.path.join(image_root_path, real_sub_path, real_videos[idx])
        # 如果目录不存在，则创建
        if not os.path.isdir(temp_save_root_path):
            os.mkdir(temp_save_root_path)
        # 保存提取的面部图像
        for j, index in enumerate(face_index):
            cv2.imwrite(os.path.join(temp_save_root_path, "%04d.png" % index), face_images[j])

        # 处理伪造视频，以下步骤对于不同技术生成的伪造视频重复
        reader = cv2.VideoCapture(os.path.join(video_root_path, deepfake_sub_path, fake_videos[idx]))
        frames_num = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
        temp_save_root_path = os.path.join(image_root_path, deepfake_sub_path, fake_videos[idx])
        if not os.path.isdir(temp_save_root_path):

            os.mkdir(temp_save_root_path)
        for j in range(frames_num):
            success, frame = reader.read()
            # 调整帧尺寸以匹配最小高度和宽度
            frame_shape = frame.shape
            if smallest_h != frame_shape[0]:
                diff = frame_shape[0] - smallest_h
                m = diff // 2
                frame = frame[m:-m, :, :]
            if smallest_w != frame_shape[1]:
                diff = frame_shape[1] - smallest_w
                m = diff // 2
                frame = frame[:, m:-m, :]
            if not success:
                break
            # 如果当前帧编号存在于面部索引中，则保存对应的面部图像
            if j in face_index:
                ymin, ymax, xmin, xmax = face_boxes[face_index.index(j)]
                face = frame[ymin:ymax, xmin:xmax, :]
                cv2.imwrite(os.path.join(temp_save_root_path, "%04d.png" % j), face)

        # 为Face2Face技术生成的伪造视频设置视频读取器
        reader = cv2.VideoCapture(
            os.path.join(video_root_path, deepfake_sub_path.replace('Deepfakes', 'Face2Face'), fake_videos[idx]))
        # 获取视频的总帧数
        f2f_frames_num = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
        # 设置用于保存面部图像的目录路径
        temp_save_root_path = os.path.join(image_root_path, deepfake_sub_path.replace('Deepfakes', 'Face2Face'),
                                           fake_videos[idx])
        # 如果保存目录不存在，则创建目录
        if not os.path.isdir(temp_save_root_path):
            os.mkdir(temp_save_root_path)
        # 遍历视频的每一帧
        for j in range(f2f_frames_num):
            # 读取一帧
            success, frame = reader.read()
            # 获取帧的尺寸
            frame_shape = frame.shape
            # 调整帧的高度以匹配最小高度
            if smallest_h != frame_shape[0]:
                diff = frame_shape[0] - smallest_h
                m = diff // 2
                frame = frame[m:-m, :, :]
            # 调整帧的宽度以匹配最小宽度
            if smallest_w != frame_shape[1]:
                diff = frame_shape[1] - smallest_w
                m = diff // 2
                frame = frame[:, m:-m, :]
            # 如果读取失败，则停止处理
            if not success:
                break
            # 如果当前帧编号在面部索引中，提取并保存面部图像
            if j in face_index:
                ymin, ymax, xmin, xmax = face_boxes[face_index.index(j)]
                face = frame[ymin:ymax, xmin:xmax, :]
                cv2.imwrite(os.path.join(temp_save_root_path, "%04d.png" % j), face)

        # 下面的代码段重复上述过程，用于处理NeuralTextures技术生成的伪造视频
        reader = cv2.VideoCapture(
            os.path.join(video_root_path, deepfake_sub_path.replace('Deepfakes', 'NeuralTextures'), fake_videos[idx]))
        frames_num = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
        temp_save_root_path = os.path.join(image_root_path,
                                           deepfake_sub_path.replace('Deepfakes', 'NeuralTextures'),
                                           fake_videos[idx])
        if not os.path.isdir(temp_save_root_path):
            os.mkdir(temp_save_root_path)
        for j in range(frames_num):
            success, frame = reader.read()
            frame_shape = frame.shape
            if smallest_h != frame_shape[0]:
                diff = frame_shape[0] - smallest_h
                m = diff // 2
                frame = frame[m:-m, :, :]
            if smallest_w != frame_shape[1]:
                diff = frame_shape[1] - smallest_w
                m = diff // 2
                frame = frame[:, m:-m, :]
            if not success:
                break
            if j in face_index:
                ymin, ymax, xmin, xmax = face_boxes[face_index.index(j)]
                face = frame[ymin:ymax, xmin:xmax, :]
                cv2.imwrite(os.path.join(temp_save_root_path, "%04d.png" % j), face)

        reader = cv2.VideoCapture(
            os.path.join(video_root_path, deepfake_sub_path.replace('Deepfakes', 'FaceSwap'), fake_videos[idx]))
        frames_num = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
        temp_save_root_path = os.path.join(image_root_path, deepfake_sub_path.replace('Deepfakes', 'FaceSwap'),
                                           fake_videos[idx])
        if not os.path.isdir(temp_save_root_path):
            os.mkdir(temp_save_root_path)
        for j in range(frames_num):
            success, frame = reader.read()
            frame_shape = frame.shape
            if smallest_h != frame_shape[0]:
                diff = frame_shape[0] - smallest_h
                m = diff // 2
                frame = frame[m:-m, :, :]
            if smallest_w != frame_shape[1]:
                diff = frame_shape[1] - smallest_w
                m = diff // 2
                frame = frame[:, m:-m, :]

            if not success:
                break
            if j in face_index:
                ymin, ymax, xmin, xmax = face_boxes[face_index.index(j)]
                face = frame[ymin:ymax, xmin:xmax, :]
                cv2.imwrite(os.path.join(temp_save_root_path, "%04d.png" % j), face)

        #reader = cv2.VideoCapture(
        #    os.path.join(video_root_path, deepfake_sub_path.replace('Deepfakes', 'FaceShifter'), fake_videos[idx]))
        #frames_num = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
        #temp_save_root_path = os.path.join(image_root_path, deepfake_sub_path.replace('Deepfakes', 'FaceShifter'),
        #                                   fake_videos[idx])
        #if not os.path.isdir(temp_save_root_path):
        #    os.mkdir(temp_save_root_path)
        #for j in range(frames_num):
        #    success, frame = reader.read()
        #    frame_shape = frame.shape
        #    if smallest_h != frame_shape[0]:
        #        diff = frame_shape[0] - smallest_h
        #        m = diff // 2
        #        frame = frame[m:-m, :, :]
        #    if smallest_w != frame_shape[1]:
        #        diff = frame_shape[1] - smallest_w
        #        m = diff // 2
        #        frame = frame[:, m:-m, :]

        #    if not success:
        #        break
        #    if j in face_index:
        #        ymin, ymax, xmin, xmax = face_boxes[face_index.index(j)]
        #        face = frame[ymin:ymax, xmin:xmax, :]
        #        cv2.imwrite(os.path.join(temp_save_root_path, "%04d.png" % j), face)


if __name__ == "__main__":
    # 设置视频和图像数据的根目录路径
    video_root_path = '/home/lab/workspace/works/zqs/datasets/FF++'
    image_root_path = '/home/lab/workspace/works/zqs/datasets/FF++/Face'
    # 设置真实视频和伪造视频的子目录路径
    real_sub_path = 'ORIGINAL/c23/videos'
    deepfake_sub_path = 'Deepfakes/c23/videos'
    # 如果图像根目录不存在，则创建
    if not os.path.isdir(image_root_path):
        os.mkdir(image_root_path)

    # 为真实视频创建存储路径
    temp_image_root_path = image_root_path
    for name in real_sub_path.split('/'):
        temp_image_root_path = os.path.join(temp_image_root_path, name)
        if not os.path.isdir(temp_image_root_path):
            os.mkdir(temp_image_root_path)

    # 为Deepfakes伪造视频创建存储路径
    temp_image_root_path = image_root_path
    for name in deepfake_sub_path.split('/'):
        temp_image_root_path = os.path.join(temp_image_root_path, name)
        if not os.path.isdir(temp_image_root_path):
            os.mkdir(temp_image_root_path)

    # 为Face2Face伪造视频创建存储路径
    temp_image_root_path = image_root_path
    for name in deepfake_sub_path.replace('Deepfakes', 'Face2Face').split('/'):
        temp_image_root_path = os.path.join(temp_image_root_path, name)
        if not os.path.isdir(temp_image_root_path):
            os.mkdir(temp_image_root_path)

    # 为FaceSwap伪造视频创建存储路径
    temp_image_root_path = image_root_path
    for name in deepfake_sub_path.replace('Deepfakes', 'FaceSwap').split('/'):
        temp_image_root_path = os.path.join(temp_image_root_path, name)
        if not os.path.isdir(temp_image_root_path):
            os.mkdir(temp_image_root_path)

    # 为NeuralTextures伪造视频创建存储路径
    temp_image_root_path = image_root_path
    for name in deepfake_sub_path.replace('Deepfakes', 'NeuralTextures').split('/'):
        temp_image_root_path = os.path.join(temp_image_root_path, name)
        if not os.path.isdir(temp_image_root_path):
            os.mkdir(temp_image_root_path)

    # 为FaceShifter伪造视频创建存储路径
    #temp_image_root_path = image_root_path
    #for name in deepfake_sub_path.replace('Deepfakes', 'FaceShifter').split('/'):
    #    temp_image_root_path = os.path.join(temp_image_root_path, name)
    #    if not os.path.isdir(temp_image_root_path):
    #        os.mkdir(temp_image_root_path)

    # 打开测试集的JSON文件
    f = open('splits/test.json', 'r')
    # 加载测试集JSON数据
    test_json = json.load(f)
    # 初始化伪造视频和真实视频的列表
    fake_videos = []
    real_videos = []
    # 遍历测试集中的视频名称
    for video_name in test_json:
        # 生成伪造视频文件名并添加到列表
        fake_videos.append(video_name[0] + '_' + video_name[1] + '.mp4')
        fake_videos.append(video_name[1] + '_' + video_name[0] + '.mp4')
        # 生成真实视频文件名并添加到列表
        real_videos.append(video_name[0] + '.mp4')
        real_videos.append(video_name[1] + '.mp4')

    # 打开验证集的JSON文件
    f = open('splits/val.json', 'r')
    # 加载验证集JSON数据
    test_json = json.load(f)
    # 遍历验证集中的视频名称，重复上述过程
    for video_name in test_json:
        fake_videos.append(video_name[0] + '_' + video_name[1] + '.mp4')
        fake_videos.append(video_name[1] + '_' + video_name[0] + '.mp4')
        real_videos.append(video_name[0] + '.mp4')
        real_videos.append(video_name[1] + '.mp4')

    # 打开训练集的JSON文件
    f = open('splits/train.json', 'r')
    # 加载训练集JSON数据
    test_json = json.load(f)
    # 遍历训练集中的视频名称，重复上述过程
    for video_name in test_json:
        fake_videos.append(video_name[0] + '_' + video_name[1] + '.mp4')
        fake_videos.append(video_name[1] + '_' + video_name[0] + '.mp4')
        real_videos.append(video_name[0] + '.mp4')
        real_videos.append(video_name[1] + '.mp4')

    # 调用main函数，传入视频根路径、图像根路径、真实视频子路径、深度伪造视频子路径、真实视频列表和伪造视频列表
    main(video_root_path, image_root_path, real_sub_path, deepfake_sub_path, real_videos, fake_videos)

