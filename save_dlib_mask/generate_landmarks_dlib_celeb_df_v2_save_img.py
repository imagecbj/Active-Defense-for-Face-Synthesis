from tqdm import tqdm
from generate_landmarks_dlib import save_landmarks
import argparse

from functools import partial
from multiprocessing.pool import Pool

import cv2
cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

import os
# 设置环境变量以控制线程数
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

def parse_args():
    # 创建命令行解析器
    parser = argparse.ArgumentParser(description='Extract face from videos')
    # 添加命令行参数，指定图像根路径
    parser.add_argument('--image_root_path', type=str,
                        default='/home/lab/workspace/works/zqs/datasets/Celeb_DF/Face/Celeb-real')
    # 解析命令行参数
    args = parser.parse_args()
    return args

def main():
    # 解析命令行参数
    args = parse_args()
    # 从命令行参数获取图像根路径
    image_root_path = args.image_root_path
    # 初始化输入目录列表
    input_dir = []

    # 遍历图像根路径下的所有文件/文件夹
    for index, video in tqdm(enumerate(os.listdir(image_root_path))):
        # 将每个文件/文件夹的完整路径加入到输入目录列表
        input_dir.append(os.path.join(image_root_path, video))
    # 设置保存目录
    save_dir = os.path.join(image_root_path, 'dlib_landmarks')
    # 如果保存目录不存在，则创建
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    # 打印输入目录的数量
    print(len(input_dir))
    # 打印CPU核心数
    print('cpu_count: %d' % os.cpu_count())
    # 创建进程池，进程数为CPU核心数的三分之一
    with Pool(processes=int(os.cpu_count()/3)) as p:
        # 创建进度条
        with tqdm(total=len(input_dir)) as pbar:
            # 创建部分应用函数，预设保存目录参数
            func = partial(save_landmarks, save_dir=save_dir)
            # 并行处理输入目录，每处理一个目录，进度条更新一次
            for v in p.imap_unordered(func, input_dir):
                pbar.update()

# 如果此脚本作为主程序运行，则执行main函数
if __name__ == "__main__":
    main()




















