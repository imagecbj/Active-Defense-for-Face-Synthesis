"""
Author: HanChen
Date: 15.10.2020
"""
import numpy as np
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import argparse
import random
from random import randint
import pandas as pd
from tqdm import tqdm
from kornia import color
import lpips
from facenet_pytorch import InceptionResnetV1
from model import ENet,  dsl_net
from model_2 import ENet2, RNet2
from torchvision.utils import save_image
from wm_net import *
from transforms import build_transforms
from utils import AverageMeter, psnr, gauss_noise, ssim,quantization
from utils import decoded_message_error_rate
from utils import message_expand,DWT,IWT, decoded_message_error_rate_c
from ff_df import ff_df_Dataloader
import torch.nn.functional as F
from wm_net import embed_watermark,extract_watermark
from noise_layers.crop import Crop
from noise_layers.resize import Resize
from losses import Arcface_loss
import Noise as Noise
import os
from datetime import datetime

######################################################################


# Save model
def save_network(network, save_filename):
    torch.save(network.cpu().state_dict(), save_filename)
    if torch.cuda.is_available():
        network.cuda()


def load_network(network, save_filename):
    network.load_state_dict(torch.load(save_filename))
    return network

def lr_decay(lr, epoch, opt):
    lr = lr * (0.1 ** (epoch // 2))
    for param_group in opt.param_groups:
        param_group['lr'] = lr

def parse_args():
    parser = argparse.ArgumentParser(description='Training network on ffdf dataset')


    # parser.add_argument('--pretrained', type=str, default='/home/linyz/UCL_Blending/xception-43020ad28.pth')
    parser.add_argument('--face_id_save_path', type=str, default='/data/linyz/SIDT/Arcface.pth')

    parser.add_argument('--gpu_id', type=int, default=1)

    parser.add_argument('--rnd_bri_ramp', type=int, default=5000)
    parser.add_argument('--rnd_hue_ramp', type=int, default=5000)
    parser.add_argument('--jpeg_quality_ramp', type=float, default=5000)
    parser.add_argument('--rnd_noise_ramp', type=int, default=5000)
    parser.add_argument('--contrast_ramp', type=int, default=5000)
    parser.add_argument('--rnd_sat_ramp', type=int, default=5000)
    parser.add_argument('--rnd_crop_ramp', type=int, default=5000)  # Borrowed from HiDDeN
    parser.add_argument('--rnd_resize_ramp', type=int, default=5000)  # Borrowed from HiDDeN
    parser.add_argument('--rnd_bri', type=float, default=.1)
    parser.add_argument('--rnd_hue', type=float, default=.05)
    parser.add_argument('--jpeg_quality', type=float, default=50)
    parser.add_argument('--rnd_noise', type=float, default=.02)
    parser.add_argument('--contrast_low', type=float, default=.8)
    parser.add_argument('--contrast_high', type=float, default=1.2)
    parser.add_argument('--rnd_sat', type=float, default=0.5)
    parser.add_argument('--blur_prob', type=float, default=0.1)
    parser.add_argument('--no_jpeg', type=bool, default=True)
    parser.add_argument('--is_cuda', type=bool, default=False)
    parser.add_argument('--rnd_crop', type=float, default=0.2)  # Borrowed from HiDDeN
    parser.add_argument('--rnd_resize', type=float, default=0.2)  # Borrowed from HiDDeN

    parser.add_argument('--message_length', type=int, default=32)
    parser.add_argument('--message_range', type=float, default=0.1)
    parser.add_argument('--use_noise', type=bool, default=True)
    # parser.add_argument('--num_epochs', type=int, default=1000)

    parser.add_argument('--y_scale', type=float, default=1.0)
    parser.add_argument('--u_scale', type=float, default=10.0)
    parser.add_argument('--v_scale', type=float, default=10.0)
    parser.add_argument('--train', type=int, default=0)
    parser.add_argument('--val', type=int, default=1)

    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--val_epochs', type=int, default=20)
    parser.add_argument('--start_val_epochs', type=int, default=50)
    parser.add_argument('--adjust_lr_epochs', type=int, default=50)
    parser.add_argument('--base_lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--resolution', type=int, default=256)
    parser.add_argument('--val_batch_size', type=int, default=4)
    parser.add_argument('--alpha', type=list, default=[1, 1])
    # parser.add_argument('--root_path', type=str, default='/home/lab/workspace/works/zqs/datasets/FF++')
    parser.add_argument('--root_path', type=str, default='/home/lab/workspace/works/zqs/datasets/Celeb_DF')
    parser.add_argument('--save_path', type=str, default='./save_results')

    args = parser.parse_args()
    return args
def main():
    args = parse_args()

    transform_train, transform_test = build_transforms(args.resolution, args.resolution,
                                                       max_pixel_value=255.0, norm_mean=[0, 0, 0],
                                                       norm_std=[1.0, 1.0, 1.0])
    yuv_scales = torch.Tensor([args.y_scale, args.u_scale, args.v_scale]).cuda()
    with open('./save_txt/CDF/train_cdf_fake.txt', 'r') as f:
        fake_train_videos = f.readlines()
        fake_train_videos = [i.strip() for i in fake_train_videos]
    with open('./save_txt/CDF/train_cdf_real.txt', 'r') as f:
        source_train_videos = f.readlines()
        source_train_videos = [i.strip() for i in source_train_videos]

    #
    with open('./save_txt/CDF/test_cdf_fake.txt', 'r') as f:
        fake_val_videos = f.readlines()
        fake_val_videos = [i.strip() for i in fake_val_videos]

    with open('./save_txt/CDF/test_cdf_real.txt', 'r') as f:
        source_val_videos = f.readlines()
        source_val_videos = [i.strip() for i in source_val_videos]

    # with open('./save_txt/Deepfakes/c23/train_df_fake_c23.txt', 'r') as f:
    #     fake_train_videos = f.readlines()
    #     fake_train_videos = [i.strip() for i in fake_train_videos]
    # #
    # with open('./save_txt/Deepfakes/c23/train_df_real_c23.txt', 'r') as f:
    #     source_train_videos = f.readlines()
    #     source_train_videos = [i.strip() for i in source_train_videos]
    #
    #
    # with open('./save_txt/Deepfakes/c23/val_df_fake_c23.txt', 'r') as f:
    #     fake_val_videos = f.readlines()
    #     fake_val_videos = [i.strip() for i in fake_val_videos]
    #
    # with open('./save_txt/Deepfakes/c23/val_df_real_c23.txt', 'r') as f:
    #     source_val_videos = f.readlines()
    #     source_val_videos = [i.strip() for i in source_val_videos]

    train_dataset = ff_df_Dataloader(root_path=args.root_path, fake_video_names=fake_train_videos,
                                     source_video_names=source_train_videos,
                                     phase='valid', transform=transform_train,test_frame_nums=10, size=(args.resolution, args.resolution))

    val_dataset = ff_df_Dataloader(root_path=args.root_path, fake_video_names=fake_val_videos,
                                   source_video_names=source_val_videos,
                                   phase='valid', transform=transform_test, test_frame_nums=20,
                                   size=(args.resolution, args.resolution))

    message_range = args.message_range
    message_length = args.message_length

    torch.autograd.set_detect_anomaly(True)

    print('Test Images Number: %d' % len(val_dataset))
    # print('All Train videos Number: %d' % (len(fake_train_videos)))
    print('Use Train videos Number: %d' % len(train_dataset))

    dwt = DWT()
    iwt = IWT()

    # 初始化编码器模型，并将其移至GPU上
    encoder2 = ENet2().cuda()
    file_path = '/home/lab/workspace/works/zqs/project/Face-Recover/save_results/models/encoder2.pth'
    if os.path.exists(file_path):
        state_dicts = torch.load(file_path)
        # Source_encoder = torch.nn.DataParallel(Source_encoder)
        # new_state_dict = {'module.' + k: v for k, v in state_dicts.items()}
        encoder2.load_state_dict(state_dicts)
        try:
            optim.load_state_dict(state_dicts['opt'])
        except:
            print('en2 Cannot load optimizer for some reason or other')

    decoder2 = RNet2().cuda()
    file_path = '/home/lab/workspace/works/zqs/project/Face-Recover/save_results/models/decoder2.pth'
    if os.path.exists(file_path):
        state_dicts = torch.load(file_path)
        # Fake_encoder = torch.nn.DataParallel(Fake_encoder)
        # new_state_dict = {'module.' + k: v for k, v in state_dicts.items()}
        decoder2.load_state_dict(state_dicts)
        try:
            optim.load_state_dict(state_dicts['opt'])
        except:
            print('de2 Cannot load optimizer for some reason or other')


    encoder = encoder2
    decoder = decoder2

    # 初始化编码器模型，并将其移至GPU上

    best_loss = 100
    # 为编码器模型创建Adam优化器
    optimizer_encoder = optim.Adam(filter(lambda p: p.requires_grad,
                                   encoder.parameters()), lr=args.base_lr)
    optimizer_decoder = optim.Adam(filter(lambda p: p.requires_grad,
                                   decoder.parameters()), lr=args.base_lr)

    scheduler_encoder = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_encoder, T_max=args.adjust_lr_epochs)
    scheduler_decoder = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_decoder, T_max=args.adjust_lr_epochs)

    # 获取当前时间，格式为 年-月-日-时-分
    current_time = datetime.now().strftime('%Y-%m-%d-%H-%M')

    # 构建文件保存路径，包含当前时间
    csv_dir = os.path.join(args.save_path, 'report', current_time)
    os.makedirs(csv_dir, exist_ok=True)  # 确保目录存在，如果不存在则创建

    # 创建训练数据加载器
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True,
                                               shuffle=True, num_workers=4, pin_memory=True)
    # 创建验证数据加载器
    val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=True, batch_size=args.val_batch_size,
                                             drop_last=True, num_workers=4, pin_memory=True)


    # 初始化全局步骤计数器
    global_step = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #图片测试

    # 遍历所有的训练周期
    for epoch in range(args.num_epochs):
        print('Epoch {}/{}'.format(epoch, args.num_epochs - 1))  # 打印当前训练周期
        print('-' * 10)  # 打印分隔线

        # WM_NET.train(True)  # 将编码器设置为训练模式
        encoder.train(True)
        decoder.train(True)
        # Fake_encoder.train(True)  # 将解码器设置为训练模式
        # 初始化用于记录各种损失的计数器
        Hiding_l2losses = AverageMeter()
        Revealed_l2losses = AverageMeter()
        BitACC = AverageMeter()
        Sum_losses = AverageMeter()
        training_process = tqdm(train_loader, ncols=130)  # 使用tqdm创建一个进度条
        psnr1 = 0
        ACC = 0

        if args.train == 1:
            # 遍历训练数据加载器中的数据
            for idx, (fake_image, source_image, face_mask) in enumerate(training_process):

                # 获取message张量的形状
                fake_shape = fake_image.shape

                zero_tensor = torch.zeros(fake_shape,device = 'cuda')
                if idx > 0:
                    # 更新进度条的描述信息，显示各种损失的平均值
                    training_process.set_description(
                        'Ep:%d H_m:%.4e, R_m:%.4e, PSNR1:%.4f,ACC:%.4f Sum:%.4e' %
                        (epoch, Hiding_l2losses.avg.item(),Revealed_l2losses.avg.item(),psnr1_avg,ACC_avg,Sum_losses.avg.item()))
                optimizer_encoder.zero_grad()  # 清零编码器的梯度
                optimizer_decoder.zero_grad()  # 清零编码器的梯度
                # optimizerR.zero_grad()  # 清零解码器的梯度


                if 1:
                    # 将图像数据和面具数据移动到GPU，并将它们包装成Variable
                    fake_image = Variable(fake_image.cuda().detach())
                    source_image = Variable(source_image.cuda().detach())
                    face_mask = Variable(face_mask.cuda().detach())


                    for param_group in optimizer_encoder.param_groups:
                        cur_lr = param_group['lr']

                    # message0 = torch.randint(0, 2, (1, message_length), dtype=torch.float32).to(device)
                    message0 = torch.randint(0, 2, (fake_image.shape[0], 256), dtype=torch.float32).to(device)


                    message = message0
                    message1 = message.view(-1,args.message_length,args.message_length)


                    fake_images_with_wm = encoder(fake_image, message)

                    # container_fake_image_jpeg = Noise.jpeg_compression_train(fake_images_with_wm)
                    container_fake_image_noise = Noise.noise(fake_images_with_wm,args.use_noise)

                    rev_message = decoder(container_fake_image_noise)

                    ACC_0 = 100 * (1 - decoded_message_error_rate_c(message0, rev_message.cuda(), message_length,
                                                                           args.batch_size))

                    psnr1_0 = psnr(fake_image[0].cpu(), fake_images_with_wm[0].cpu())
                    # psnr1_0 = psnr(color.yuv_to_rgb(test_image_yuv_2)[0].cpu(), color.yuv_to_rgb(fake_images_with_wm)[0].cpu())
                    psnr1 = psnr1 + psnr1_0
                    psnr1_avg = psnr1 / (idx + 1)
                    ACC = ACC + ACC_0
                    ACC_avg = ACC / (idx + 1)

                    lossR = 0
                    lossH = 0

                    for i in range(args.batch_size):
                        lossH += F.mse_loss(fake_image, fake_images_with_wm)
                        lossR += F.mse_loss(rev_message, message)


                    Hiding_l2losses.update(lossH.cpu(), fake_image.size(0))
                    Revealed_l2losses.update(lossR.cpu(), message.size(0))
                    BitACC.update(ACC_0)

                    loss = args.alpha[0] * lossH + args.alpha[1] * lossR
                    Sum_losses.update(loss.cpu(), fake_image.size(0))
                    loss.backward()
                    # optimizerR.step()
                    optimizer_encoder.step()
                    optimizer_decoder.step()
                    lr_decay(cur_lr, epoch, optimizer_encoder)
                    lr_decay(cur_lr, epoch, optimizer_decoder)

            # 创建一个空的DataFrame用于记录本次验证的各项指标
            df_acc = pd.DataFrame()
            # 添加各项指标到DataFrame
            df_acc['epoch'] = [epoch]
            df_acc['H_loss'] = [round(Hiding_l2losses.avg.item(),4)]
            df_acc['R_loss'] = [round(Revealed_l2losses.avg.item(),4)]
            df_acc['BitACC'] = [round(BitACC.avg.item(),4)]
            df_acc['LearningRate'] = [cur_lr]
            # df_acc['Revealed_idlosses'] = [Revealed_idlosses.avg.item()]
            df_acc['Sum_Loss'] = [round(Sum_losses.avg.item(),4)]

            # 根据当前是哪个周期决定是否在CSV文件中包含头部信息
            if epoch == 0:
                df_acc.to_csv('%s/train.csv' % csv_dir, mode='a', index=None)
            else:
                df_acc.to_csv('%s/train.csv' % csv_dir, mode='a', index=None, header=None)


            if (best_loss > Sum_losses.avg.item()) or ((epoch + 1) % 5 == 0) :
                best_loss = Sum_losses.avg.item()
                save_network(encoder, '%s/models/encoder.pth' % (args.save_path ))
                save_network(decoder, '%s/models/decoder.pth' % (args.save_path ))


        # # 如果达到了指定的验证周期并且已经超过了开始验证的起始周期
        if (epoch + 1) % args.val_epochs == 0 and epoch > args.start_val_epochs and args.val == 1:
            # 将模型设置为评估模式
            encoder.train(False)
            decoder.train(False)
            encoder.eval()
            decoder.eval()

            # 初始化用于记录验证过程中各种损失的计数器
            Hiding_l2losses = AverageMeter()
            Revealed_l2losses = AverageMeter()
            BitACC = AverageMeter()
            Sum_losses = AverageMeter()
            psnr1 = 0
            ACC = 0

        #     # 使用tqdm创建一个验证数据的进度条
            valid_process = tqdm(val_loader,ncols=130)

            # 对于验证过程中的每一批数据
            for idx, (fake_image, source_image, face_mask) in enumerate(valid_process):

                # 获取message张量的形状
                fake_shape = fake_image.shape

                # 使用获取的形状初始化一个全0张量
                zero_tensor = torch.zeros(fake_shape, device='cuda')
                # 更新验证进度的描述
                if idx > 0:
                    valid_process.set_description(
                        'Ep:%d H_m:%.4e, R_m:%.4e, PSNR1:%.4f,ACC:%.4f Sum:%.4e' %
                    (epoch, Hiding_l2losses.avg.item(),Revealed_l2losses.avg.item(),psnr1_avg,ACC_avg,Sum_losses.avg.item()))


                # 在无梯度模式下进行验证
                with torch.no_grad():
                    # 将图像数据和面具数据移动到GPU，并将它们包装成Variable
                    fake_image = Variable(fake_image.cuda().detach())
                    source_image = Variable(source_image.cuda().detach())
                    face_mask = Variable(face_mask.cuda().detach())
                    message0 = torch.randint(0, 2, (fake_image.shape[0], 256), dtype=torch.float32).to(device)

                    message = message0
                    message1 = message.view(-1, args.message_length, args.message_length)

                    fake_images_with_wm = encoder(fake_image, message)

                    # container_fake_image_jpeg = Noise.jpeg_compression_train(fake_images_with_wm)
                    container_fake_image_noise = Noise.noise(fake_images_with_wm, args.use_noise)
                    rev_message = decoder(container_fake_image_noise)

                    ACC_0 = 100 * (1 - decoded_message_error_rate_c(message0, rev_message.cuda(), message_length,
                                                                    args.batch_size))

                    psnr1_0 = psnr(fake_image[0].cpu(), fake_images_with_wm[0].cpu())
                    # psnr1_0 = psnr(color.yuv_to_rgb(test_image_yuv_2)[0].cpu(), color.yuv_to_rgb(fake_images_with_wm)[0].cpu())
                    psnr1 = psnr1 + psnr1_0
                    psnr1_avg = psnr1 / (idx + 1)
                    ACC = ACC + ACC_0
                    ACC_avg = ACC / (idx + 1)

                    lossR = 0
                    lossH = 0

                    for i in range(args.batch_size):
                        lossH += F.mse_loss(fake_image, fake_images_with_wm)
                        lossR += F.mse_loss(rev_message, message)

                    Hiding_l2losses.update(lossH.cpu(), fake_image.size(0))
                    Revealed_l2losses.update(lossR.cpu(), message.size(0))
                    BitACC.update(ACC_0)

                    loss = args.alpha[0] * lossH + args.alpha[1] * lossR

                    Sum_losses.update(loss.cpu(), fake_image.size(0))



            # 创建一个空的DataFrame用于记录本次验证的各项指标
            df_acc = pd.DataFrame()
            # 添加各项指标到DataFrame
            df_acc['epoch'] = [epoch]
            df_acc['H_loss'] = [round(Hiding_l2losses.avg.item(),4)]
            # df_acc['Hiding_idlosses'] = [Hiding_idlosses.avg.item()]
            df_acc['R_loss'] = [round(Revealed_l2losses.avg.item(),4)]
            df_acc['BitACC'] = [round(BitACC.avg.item(),4)]
            df_acc['Sum_Loss'] = [round(Sum_losses.avg.item(),4)]

            # 根据当前是哪个周期决定是否在CSV文件中包含头部信息
            if epoch  != (args.val_epochs + args.start_val_epochs):
                df_acc.to_csv('%s/validation.csv' % csv_dir, mode='a', index=None, header=None)
            else:
                df_acc.to_csv('%s/validation.csv' % csv_dir, mode='a', index=None)

            # # 如果当前损失小于之前记录的最佳损失，则更新最佳损失并保存当前的网络模型
            # if best_loss > Sum_losses.avg.item():
            #     best_loss = Sum_losses.avg.item()
            #     save_network(encoder, '%s/models/MBRS_EN.pth' % args.save_path)
            #     # save_network(Fake_encoder, '%s/models/Fake_encoder.pth' % args.save_path)
            #     save_network(decoder, '%s/models/best/MBRS_DE%s.pth' % (args.save_path, epoch))  # 保存编码器模型
            #     # save_network(Fake_encoder, '%s/models/best/Fake_encoder_%s.pth' % (args.save_path, epoch))  # 保存解码器模型
        scheduler_encoder.step()
        scheduler_decoder.step()
#
if __name__ == "__main__":
    args = parse_args()  # 解析命令行参数
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)  # 设置CUDA可见设备
    # 如果保存路径不存在，则创建它
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    # 如果模型保存路径不存在，则创建它
    if not os.path.exists('%s/models' % args.save_path):
        os.makedirs('%s/models' % args.save_path)
    # 如果报告保存路径不存在，则创建它
    if not os.path.exists('%s/report' % args.save_path):
        os.makedirs('%s/report' % args.save_path)
    # 如果图像保存路径不存在，则创建它
    if not os.path.exists('%s/images' % args.save_path):
        os.makedirs('%s/images' % args.save_path)
    if not os.path.exists('%s/models/best' % args.save_path):
        os.makedirs('%s/models/best' % args.save_path)
    if not os.path.exists('%s/models/train' % args.save_path):
        os.makedirs('%s/models/train' % args.save_path)
    main()  # 执行主函数

