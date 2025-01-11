import os
import json
import random
from tqdm import tqdm


def ff_main():
    # 定义真实视频和伪造视频的路径
    real_path = '/data1/zqs/datasets/FF++/ORIGINAL/c40/videos'
    fake_path = '/data1/zqs/datasets/FF++/Deepfakes/c40/videos'
    # 读取测试集视频名称
    f = open('splits/test.json', 'r')
    test_json = json.load(f)
    test_videos = []
    # 将测试集中的伪造视频和真实视频的路径添加到列表
    for video_name in test_json:
        # 添加伪造视频路径
        input_video_path = os.path.join(fake_path, video_name[0] + '_' + video_name[1] + '.mp4')
        test_videos.append(input_video_path)
        input_video_path = os.path.join(fake_path, video_name[1] + '_' + video_name[0] + '.mp4')
        test_videos.append(input_video_path)
        # 添加真实视频路径
        input_video_path = os.path.join(real_path, video_name[0] + '.mp4')
        test_videos.append(input_video_path)
        input_video_path = os.path.join(real_path, video_name[1] + '.mp4')
        test_videos.append(input_video_path)

    # 重复上述步骤，读取训练集视频名称
    f = open('splits/train.json', 'r')
    train_json = json.load(f)
    train_videos = []
    for video_name in train_json:
        input_video_path = os.path.join(fake_path, video_name[0] + '_' + video_name[1] + '.mp4')
        train_videos.append(input_video_path)
        input_video_path = os.path.join(fake_path, video_name[1] + '_' + video_name[0] + '.mp4')
        train_videos.append(input_video_path)
        input_video_path = os.path.join(real_path, video_name[0] + '.mp4')
        train_videos.append(input_video_path)
        input_video_path = os.path.join(real_path, video_name[1] + '.mp4')
        train_videos.append(input_video_path)

    # 重复上述步骤，读取验证集视频名称
    f = open('splits/val.json', 'r')
    val_json = json.load(f)
    val_videos = []
    for video_name in val_json:
        input_video_path = os.path.join(fake_path, video_name[0] + '_' + video_name[1] + '.mp4')
        val_videos.append(input_video_path)
        input_video_path = os.path.join(fake_path, video_name[1] + '_' + video_name[0] + '.mp4')
        val_videos.append(input_video_path)
        input_video_path = os.path.join(real_path, video_name[0] + '.mp4')
        val_videos.append(input_video_path)
        input_video_path = os.path.join(real_path, video_name[1] + '.mp4')
        val_videos.append(input_video_path)

    # 创建文本文件，用于保存训练集、验证集和测试集的视频路径
    train_txt = open('./save_txt/train_ff.txt', 'w')
    val_txt = open('./save_txt/val_ff.txt', 'w')
    test_txt = open('./save_txt/test_ff.txt', 'w')
    # 将训练集视频路径写入文件
    for i in range(len(train_videos)):
        train_txt.write(train_videos[i] + '\n')
    # 将验证集视频路径写入文件
    for i in range(len(val_videos)):
        val_txt.write(val_videos[i] + '\n')
    # 将测试集视频路径写入文件
    for i in range(len(test_videos)):
        test_txt.write(test_videos[i] + '\n')



def cdf_main():
    # 定义真实视频和伪造视频的路径
    real_path = 'Celeb-real'
    fake_path = 'faceshifter'
    # 读取测试集视频名称
    f = open('splits/cdfsh_test.json', 'r')
    test_json = json.load(f)
    test_fake_videos = []
    test_real_videos = []
    # 将测试集中的伪造视频和真实视频的路径添加到列表
    for video_name in test_json:
        # 添加伪造视频路径
        input_video_path = os.path.join(fake_path, video_name[0] + '_' + video_name[1] + '_' + video_name[2] + '.mp4')
        test_fake_videos.append(input_video_path)
        # input_video_path = os.path.join(fake_path, video_name[1] + '_' + video_name[0] + '.mp4')
        # test_fake_videos.append(input_video_path)
        # 添加真实视频路径
        input_video_path = os.path.join(real_path, video_name[0] + '_' + video_name[2] + '.mp4')
        test_real_videos.append(input_video_path)
        # input_video_path = os.path.join(real_path, video_name[1] + '.mp4')
        # test_real_videos.append(input_video_path)

    # 重复上述步骤，读取训练集视频名称
    f = open('splits/cdfsh_train.json', 'r')
    train_json = json.load(f)
    train_fake_videos = []
    train_real_videos = []
    for video_name in train_json:
        # 添加伪造视频路径
        input_video_path = os.path.join(fake_path, video_name[0] + '_' + video_name[1] + '_' + video_name[2] + '.mp4')
        train_fake_videos.append(input_video_path)
        # input_video_path = os.path.join(fake_path, video_name[1] + '_' + video_name[0] + '.mp4')
        # test_fake_videos.append(input_video_path)
        # 添加真实视频路径
        input_video_path = os.path.join(real_path, video_name[0] + '_' + video_name[2] + '.mp4')
        train_real_videos.append(input_video_path)
        # input_video_path = os.path.join(real_path, video_name[1] + '.mp4')
        # test_real_videos.append(input_video_path)

    # 创建文本文件，用于保存训练集、验证集和测试集的视频路径
    train_fake_txt = open('./save_txt/CDFSH/train_cdfsh_fake.txt', 'w')
    train_real_txt = open('./save_txt/CDFSH/train_cdfsh_real.txt','w')
    test_fake_txt = open('./save_txt/CDFSH/test_cdfsh_fake.txt', 'w')
    test_real_txt = open('./save_txt/CDFSH/test_cdfsh_real.txt', 'w')
    # 将训练集视频路径写入文件
    for i in range(len(train_fake_videos)):
        train_fake_txt.write(train_fake_videos[i] + '\n')
    for i in range(len(train_real_videos)):
        train_real_txt.write(train_real_videos[i] + '\n')

    # 将测试集视频路径写入文件
    for i in range(len(test_fake_videos)):
        test_fake_txt.write(test_fake_videos[i] + '\n')
    for i in range(len(test_real_videos)):
        test_real_txt.write(test_real_videos[i] + '\n')

# 如果这是主程序，则执行cdf_main函数
if __name__ == '__main__':
    cdf_main()
    # ff_main()

