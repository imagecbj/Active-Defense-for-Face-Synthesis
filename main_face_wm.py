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
import pandas as pd
from tqdm import tqdm
from kornia import color
import lpips
from facenet_pytorch import InceptionResnetV1
from model import ENet,  dsl_net
# from invertible_net import Inveritible_Source, Inveritible_Fake, Inveritible_Decolorization
from invertible_net import Model_att
from model_2 import ENet2, RNet2
from torchvision.utils import save_image
import Noise as Noise
from transforms import build_transforms
from utils import AverageMeter, psnr, gauss_noise, ssim,quantization
from utils import *
from utils import message_expand,DWT,IWT, decoded_message_error_rate_c
from ff_df import ff_df_Dataloader
import torch.nn.functional as F
from wm_net import embed_watermark,extract_watermark
from noise_layers.crop import Crop
from noise_layers.resize import Resize
from losses import Arcface_loss

import os


def gauss_noise(shape):
    noise = torch.zeros(shape).cuda()
    for i in range(noise.shape[0]):
        noise[i] = torch.randn(noise[i].shape).cuda()

    return noise
def save_network(network, save_filename):
    torch.save(network.cpu().state_dict(), save_filename)
    if torch.cuda.is_available():
        network.cuda()


def load_network(network, save_filename):
    network.load_state_dict(torch.load(save_filename))
    return network


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
    parser.add_argument('--wm', type=int, default=0)
    parser.add_argument('--noise', type=int, default=1)
    parser.add_argument('--choose_data', type=int, default=1)
    # parser.add_argument('--root_path', type=str, default='/home/lab/workspace/works/zqs/datasets/FF++')
    parser.add_argument('--root_path', type=str, default='/home/lab/workspace/works/zqs/datasets/Celeb_DF')
    parser.add_argument('--save_path', type=str, default='./save_results/F_en_att_DF')
    parser.add_argument('--savename', type=str, default='FR_cdf.png')

    parser.add_argument('--num_epochs', type=int, default=540)
    parser.add_argument('--val_epochs', type=int, default=1)
    parser.add_argument('--start_val_epochs', type=int, default=0)
    parser.add_argument('--adjust_lr_epochs', type=int, default=50)
    parser.add_argument('--base_lr', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--resolution', type=int, default=256)
    parser.add_argument('--val_batch_size', type=int, default=1)
    parser.add_argument('--alpha', type=list, default=[1, 1, 1,0.5, 0.85, 0.85, 1 , 0],
                        help='alpha for L2 Loss of Container, CosSimilarity Loss of Container ID, Lpip,\
                             L2 Loss of Revealed, CosSimilarity Loss of Revealed ID, Lpip')

    args = parser.parse_args()
    return args
def main():
    args = parse_args()

    transform_train, transform_test = build_transforms(args.resolution, args.resolution,
                                                       max_pixel_value=255.0, norm_mean=[0, 0, 0],
                                                       norm_std=[1.0, 1.0, 1.0])
    yuv_scales = torch.Tensor([args.y_scale, args.u_scale, args.v_scale]).cuda()
    if args.choose_data == 1:
        # args.root_path = args.root_path_cdf
        # args.save_path = args.save_path_CDF
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
    elif args.choose_data == 2:
        # args.root_path = args.root_path_cdf
        # args.save_path = args.save_path_CDFSH
        with open('./save_txt/CDFSH/train_cdfsh_fake.txt', 'r') as f:
            fake_train_videos = f.readlines()
            fake_train_videos = [i.strip() for i in fake_train_videos]
        with open('./save_txt/CDFSH/train_cdfsh_real.txt', 'r') as f:
            source_train_videos = f.readlines()
            source_train_videos = [i.strip() for i in source_train_videos]

        #
        with open('./save_txt/CDFSH/test_cdfsh_fake.txt', 'r') as f:
            fake_val_videos = f.readlines()
            fake_val_videos = [i.strip() for i in fake_val_videos]

        with open('./save_txt/CDFSH/test_cdfsh_real.txt', 'r') as f:
            source_val_videos = f.readlines()
            source_val_videos = [i.strip() for i in source_val_videos]
    else:
        # args.root_path = args.root_path_ff
        # args.save_path = args.save_path_DF
        with open('./save_txt/Deepfakes/c23/train_df_fake_c23.txt', 'r') as f:
            fake_train_videos = f.readlines()
            fake_train_videos = [i.strip() for i in fake_train_videos]
        #
        with open('./save_txt/Deepfakes/c23/train_df_real_c23.txt', 'r') as f:
            source_train_videos = f.readlines()
            source_train_videos = [i.strip() for i in source_train_videos]

        with open('./save_txt/Deepfakes/c23/val_df_fake_c23.txt', 'r') as f:
            fake_val_videos = f.readlines()
            fake_val_videos = [i.strip() for i in fake_val_videos]

        with open('./save_txt/Deepfakes/c23/val_df_real_c23.txt', 'r') as f:
            source_val_videos = f.readlines()
            source_val_videos = [i.strip() for i in source_val_videos]

    train_dataset = ff_df_Dataloader(root_path=args.root_path, fake_video_names=fake_train_videos,
                                     source_video_names=source_train_videos,
                                     phase='valid', transform=transform_train,test_frame_nums=1, size=(args.resolution, args.resolution))

    val_dataset = ff_df_Dataloader(root_path=args.root_path, fake_video_names=fake_val_videos,
                                   source_video_names=source_val_videos,
                                   phase='valid', transform=transform_test, test_frame_nums=2,
                                   size=(args.resolution, args.resolution))

    message_range = args.message_range
    message_length = args.message_length

    print('Test Images Number: %d' % len(val_dataset))
    # print('All Train videos Number: %d' % (len(fake_train_videos)))
    print('Use Train videos Number: %d' % len(train_dataset))

    dwt = DWT()
    iwt = IWT()

    # 初始化编码器模型，并将其移至GPU上
    Source_encoder = Model_att().cuda()
    file_path = '/home/lab/workspace/works/zqs/project/Face-Recover/save_results/models/Fake_encoder.pth'
    if os.path.exists(file_path):
        state_dicts = torch.load(file_path)
        print('Cannot load optimizer for some reason or other')
        # Source_encoder = torch.nn.DataParallel(Source_encoder)
        # network_state_dict = {k: v for k, v in state_dicts['net'].items() if 'tmp_var' not in k}
        Source_encoder.load_state_dict(state_dicts)
        try:
            optim.load_state_dict(state_dicts['opt'])
        except:
            print('Cannot load optimizer for some reason or other')

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
    optimizer_encoder = optim.Adam(filter(lambda p: p.requires_grad,
                                          encoder.parameters()), lr=args.base_lr)
    optimizer_decoder = optim.Adam(filter(lambda p: p.requires_grad,
                                          decoder.parameters()), lr=args.base_lr)

    scheduler_encoder = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_encoder, T_max=args.adjust_lr_epochs)
    scheduler_decoder = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_decoder, T_max=args.adjust_lr_epochs)

    best_loss = 10
    # 初始化解码器模型，并将其移至GPU上
    # Fake_encoder = Hinet().cuda()
    # 创建裁剪层，用于随机裁剪图像
    Crop_layer = Crop([1.0 - args.rnd_crop, 1.0], [1.0 - args.rnd_crop, 1.0])
    # 创建调整大小层，用于随机调整图像大小
    Resize_layer = Resize(1.0 - args.rnd_resize, 1.0 + args.rnd_resize)

    # 为编码器模型创建Adam优化器
    optimizerH = optim.Adam(filter(lambda p: p.requires_grad,
                                   Source_encoder.parameters()), lr=args.base_lr)
    # 为解码器模型创建Adam优化器
    # optimizerR = optim.Adam(filter(lambda p: p.requires_grad,
    #                                Fake_encoder.parameters()), lr=args.base_lr)
    optimizer_encoder = optim.Adam(filter(lambda p: p.requires_grad,
                                          encoder.parameters()), lr=args.base_lr)
    optimizer_decoder = optim.Adam(filter(lambda p: p.requires_grad,
                                          decoder.parameters()), lr=args.base_lr)

    scheduler_encoder = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_encoder, T_max=args.adjust_lr_epochs)
    scheduler_decoder = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_decoder, T_max=args.adjust_lr_epochs)

    # 为编码器设置余弦退火学习率调度器
    schedulerH = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerH, T_max=args.adjust_lr_epochs)
    # 为解码器设置余弦退火学习率调度器
    # schedulerR = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerR, T_max=args.adjust_lr_epochs)

    # 初始化交叉熵损失函数
    cross_criterion = torch.nn.CrossEntropyLoss()
    # 初始化均方误差损失函数
    L2_criterion = nn.MSELoss().cuda()
    # 初始化LPIPS损失函数，用于评估图像间的感知差异
    loss_fn_vgg = lpips.LPIPS(net='vgg').eval().cuda()
    # 初始化Arcface损失函数，用于人脸识别任务
    face_ID_criterion = Arcface_loss(pretrained=args.face_id_save_path).cuda()
    model = InceptionResnetV1(pretrained='vggface2').eval().cuda()
    # 创建训练数据加载器
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True,
                                               shuffle=True, num_workers=4, pin_memory=True)
    # 创建验证数据加载器
    val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=True, batch_size=args.val_batch_size,
                                             drop_last=True, num_workers=4, pin_memory=True)

    # 初始化最佳损失值

    # 初始化全局步骤计数器
    global_step = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 遍历所有的训练周期
    for epoch in range(args.num_epochs):
        print('Epoch {}/{}'.format(epoch, args.num_epochs - 1))  # 打印当前训练周期
        print('-' * 10)  # 打印分隔线

        Source_encoder.train(True)  # 将编码器设置为训练模式
        # Fake_encoder.train(True)  # 将解码器设置为训练模式
        # 初始化用于记录各种损失的计数器
        Hiding_l2losses = AverageMeter()
        Hiding_lpiplosses = AverageMeter()
        Bit_losses = AverageMeter()
        Hiding_idlosses = AverageMeter()
        Revealed_l2losses = AverageMeter()
        Revealed_lpiplosses = AverageMeter()
        Revealed_idlosses = AverageMeter()
        Revealed_wlosses = AverageMeter()
        Sum_losses = AverageMeter()
        training_process = tqdm(train_loader, ncols=170)  # 使用tqdm创建一个进度条
        psnr1 = 0
        psnr1_avg = 0
        psnr2_avg = 0
        psnr2 = 0
        ssim_s = 0
        ssim_f = 0
        id_sim = 0
        Error_Rate =0
        # 遍历训练数据加载器中的数据
        for idx, (fake_image, source_image, face_mask) in enumerate(training_process):

            message0 = torch.randint(0, 2, (fake_image.shape[0], 256), dtype=torch.float32).to(device)

            message = message0

            # Error_Rate_0 = 100 * decoded_message_error_rate_c(message0, message, message_length,
            #                                                   args.batch_size)
            # 获取message张量的形状
            fake_shape = fake_image.shape

            z_tensor = gauss_noise(fake_shape)
            if idx > 0:
                # 更新进度条的描述信息，显示各种损失的平均值
                training_process.set_description(
                    # "Epoch:%d H: l2 %.2e, id %.0f, lp %.2e, R: l2 %.2e, id %.0f, lp %.2e, Sum: %.2e" %
                    # (epoch, Hiding_l2losses.avg.item(), Hiding_idlosses.avg.item(), Hiding_lpiplosses.avg.item(),
                    #  Revealed_l2losses.avg.item(), Revealed_idlosses.avg.item(), Revealed_lpiplosses.avg.item(),
                    #  Sum_losses.avg.item()))
                    'Ep:%d H_m:%.2e, id: %.4f, l:%.2e, B_w:%.2e, R_m:%.2e, id: %.4f, l:%.2e, R_w:%.2e, PSNR1:%.2f PSNR2:%.2f Sum:%.2e' %
                    (epoch, Hiding_l2losses.avg.item(), Hiding_idlosses.avg.item(), Hiding_lpiplosses.avg.item(),Bit_losses.avg.item(),
                     Revealed_l2losses.avg.item(), Revealed_idlosses.avg.item(),Revealed_lpiplosses.avg.item(),Revealed_wlosses.avg.item(),
                     psnr1_avg,psnr2_avg, Sum_losses.avg.item()))
            optimizerH.zero_grad()  # 清零编码器的梯度
            # optimizerR.zero_grad()  # 清零解码器的梯度


            if args.train == 1:
                # 将图像数据和面具数据移动到GPU，并将它们包装成Variable
                fake_image = Variable(fake_image.cuda().detach())
                source_image = Variable(source_image.cuda().detach())
                face_mask = Variable(face_mask.cuda().detach())

                source_image_mask = source_image

                source_image_mask_dwt = dwt(source_image_mask)
                message_dwt = message
                fake_image_dwt = dwt(fake_image)
                source_image_dwt = dwt(source_image)
                z_tensor_dwt = dwt(z_tensor)

                source_input = torch.cat([fake_image_dwt, source_image_mask_dwt], dim=1)
                fake_image_1 = torch.cat([fake_image_dwt, z_tensor_dwt], dim=1)
                source_image_1 = torch.cat([source_image_dwt, z_tensor_dwt], dim=1)
                # 使用编码器处理源图像和伪造图像
                container_source_image = Source_encoder(source_input, rev=False)
                container_source_image_1 = container_source_image[:, :12, :, :]
                container_source_image_2 = container_source_image[:, 12:, :, :]

                container_source_image_1_low = container_source_image_1.narrow(1, 0, 3)
                fake_image_dwt_low = fake_image_dwt.narrow(1, 0, 3)

                container_source_image_1_iwt = iwt(container_source_image_1)
                container_source_image_2_iwt = iwt(container_source_image_2)

                container_source_image_en = encoder(container_source_image_1_iwt, message)
                # 计算隐藏损失
                lossH = torch.mean(
                    ((color.rgb_to_yuv(container_source_image_en) - color.rgb_to_yuv(fake_image))) ** 2,
                    axis=[0, 2, 3])  # 计算封装后和伪造图像之间的损失
                lossH = torch.dot(lossH, yuv_scales)  # 应用YUV缩放

                # lossH = 2 * F.mse_loss(container_source_image_1_iwt * face_mask, fake_image, reduction='mean')
                # + 0 * F.mse_loss(container_source_image_1_iwt * (1 - face_mask), fake_image, reduction='mean')
                lpips_lossH = torch.mean(
                    loss_fn_vgg((fake_image - 0.5) * 2.0, (container_source_image_en - 0.5) * 2.0))  # 计算LPIPS损失

                face_ID_lossH = face_ID_criterion(fake_image, container_source_image_en)  # 计算人脸ID损失
                # 更新损失计数器
                Hiding_l2losses.update(lossH.cpu(), fake_image.size(0))
                Hiding_lpiplosses.update(lpips_lossH.cpu(), fake_image.size(0))
                Hiding_idlosses.update(face_ID_lossH.cpu(), fake_image.size(0))

                # container_source_image_1_iwt = container_source_image_1_iwt.cuda()
                # container_source_image_1_iwt = quantization(container_source_image_1_en)

                # rev_message = decoder(container_source_image_en)
                # Error_Rate_0 = 100 * (1 - decoded_message_error_rate_c(message0, rev_message.cuda(), message_length,
                #                                                   args.batch_size))

                # container_source_image_1_iwt = quantization(container_source_image_en)
                container_fake_image_jpeg = Noise.jpeg_compression_train(container_source_image_en)
                container_source_image_1_iwt = Noise.noise(container_fake_image_jpeg, args.use_noise)

                # transformed_container_source_image_1, transformed_fake, transformed_source, transformed_face_mask = dsl_net(
                #     container_source_image_1_iwt, fake_image, source_image, face_mask, args, 100,
                #     Crop_layer, Resize_layer)

                transformed_container_source_image_1 = container_source_image_1_iwt
                transformed_source = source_image_mask
                transformed_face_mask = face_mask
                transformed_fake = fake_image

                transformed_container_source_image_dwt = dwt(transformed_container_source_image_1)
                transformed_source_dwt = dwt(transformed_source)
                transformed_face_mask_dwt = dwt(transformed_face_mask)

                transformed_container_source_image = torch.cat(
                    [transformed_container_source_image_dwt, z_tensor_dwt], dim=1)
                # 使用解码器获取解码后的源图像
                rev_source_image = Source_encoder(transformed_container_source_image, rev=True)
                rev_source_image_1 = rev_source_image[:, :12, :, :]
                rev_source_image_2 = rev_source_image[:, 12:, :, :]

                rev_message = decoder(transformed_container_source_image_1)

                loss_Bit = F.mse_loss(message0, rev_message, reduction='mean')

                Bit_losses.update(loss_Bit.cpu(), fake_image.size(0))

                rev_source_image_1_iwt = iwt(rev_source_image_1)
                rev_source_image_2_iwt = iwt(rev_source_image_2)

                rev_source_image_3 = rev_source_image_2_iwt * transformed_face_mask + source_image * (
                        1 - transformed_face_mask)

                # tensor_min = rev_source_image_3.min()
                # tensor_max = rev_source_image_3.max()
                # rev_source_image_3 = (rev_source_image_3 - tensor_min) / (
                #         tensor_max - tensor_min)
                # rev_source_image_3 = rev_source_image_3 * 255
                #
                # # 3. 将数据转换为整数
                # rev_source_image_3 = rev_source_image_3.int()
                #
                # # 4. 将整数数据转换回0-1范围
                # rev_source_image_3 = rev_source_image_3.float() / 255

                rev_source_image_3 = quantization(rev_source_image_3)

                Error_Rate_0 = 100 * (1 - decoded_message_error_rate_c(message0, rev_message.cuda(), message_length,
                                                                       args.batch_size))

                psnr1_0 = psnr(fake_image[0].cpu(), container_source_image_1_iwt[0].cpu())
                # psnr1_0 = psnr(container_source_image_1_iwt[0].cpu(), fake_image[0].cpu())

                psnr2_0 = psnr(transformed_source[0].cpu(), rev_source_image_3[0].cpu())
                # psnr2_0 = psnr(rev_source_image_3[0].cpu(), transformed_source[0].cpu())

                psnr1 = psnr1 + psnr1_0
                psnr2 = psnr2 + psnr2_0
                psnr1_avg = psnr1 / (idx + 1)
                psnr2_avg = psnr2 / (idx + 1)
                ssim_s_0 = ssim(source_image, rev_source_image_3)
                ssim_f_0 = ssim(fake_image, container_source_image_1_iwt)
                ssim_s = ssim_s + ssim_s_0
                ssim_f = ssim_f + ssim_f_0
                ssim_s_avg = ssim_s / (idx + 1)
                ssim_f_avg = ssim_f / (idx + 1)
                Error_Rate = Error_Rate + Error_Rate_0
                Error_Rate_avg = Error_Rate / (idx + 1)
                with torch.no_grad():
                    embedding1 = model(source_image)
                    embedding2 = model(rev_source_image_3)
                id_sim_0 = np.dot(embedding1[0].cpu(), embedding2[0].cpu()) / (
                            np.linalg.norm(embedding1[0].cpu()) * np.linalg.norm(embedding2[0].cpu()))
                id_sim = id_sim + id_sim_0
                id_sim_avg = id_sim / (idx + 1)

                if idx % 10 == 0:
                    import matplotlib.pyplot as plt
                    # 假设img_tensor是一个在GPU上的Tensor，形状为[C, H, W]或[N, C, H, W]
                    # 举个例子，我们创建一个假的Tensor用于演示
                    # img_tensor = torch.rand(3, 256, 256, device='cuda')  # 创建一个在GPU上的随机Tensor
                    # 如果Tensor形状是[N, C, H, W]，选择要显示的图片索引，这里我们假设只有一张图片

                    output_1 = torch.cat((transformed_source[0].cpu(), fake_image[0].cpu()), dim=2)
                    output_2 = torch.cat((rev_source_image_3[0].cpu(), container_source_image_1_iwt[0].cpu()),
                                         dim=2)
                    diff_s = (transformed_source - rev_source_image_3) * 5
                    diff_f = (fake_image - container_source_image_1_iwt) * 5
                    output_3 = torch.cat((diff_s[0].cpu(), diff_f[0].cpu()), dim=2)
                    utput_image_1 = torch.cat((output_1, output_2, output_3), dim=1)
                    # img_tensor = container_source_image_1_iwt[0]  # 选择第一张图片
                    # 将Tensor移动到CPU上
                    img_tensor_cpu = utput_image_1
                    # psnr1 = psnr(fake_image[0].cpu(),container_source_image_1_iwt[0].cpu())
                    # psnr2 = psnr(source_image[0].cpu(),rev_source_image_3[0].cpu())
                    # 将Tensor转换为NumPy数组
                    img_numpy = img_tensor_cpu.detach().numpy()
                    # 如果是3通道图片，调整Tensor形状为[H, W, C]
                    img_numpy = img_numpy.transpose(1, 2, 0)
                    plt.imshow(img_numpy)
                    plt.axis('off')
                    plt.suptitle(
                        f'test: error_rate: {Error_Rate_avg}, \n F&C: {psnr1_avg:.2f},ssim_hide: {ssim_f_avg:.4f},\n S&R: {psnr2_avg:.2f},ssim_rev: {ssim_s_avg:.4f},id_sim_rev: {id_sim_avg:.4f}'
                        , y=1.002)
                    plt.show()

                # 计算解码后图像与增强后源图像间的损失
                # lossR = torch.mean(
                #     ((color.rgb_to_yuv(rev_source_image) - color.rgb_to_yuv(transformed_source))) ** 2,
                #     axis=[0, 2, 3])
                # lossR = torch.dot(lossR, yuv_scales)  # 应用YUV缩放因子
                lossR = F.mse_loss(rev_source_image_2_iwt, transformed_source, reduction='mean')
                # 计算LPIPS损失和人脸ID损失
                lpips_lossR = torch.mean(
                    loss_fn_vgg((source_image_mask - 0.5) * 2.0, (rev_source_image_2_iwt - 0.5) * 2.0))
                # Error_Rate = decoded_message_error_rate(message0, rev_source_image_2_iwt,message_length, args.batch_size)

                loss_w = F.mse_loss(rev_source_image_1_iwt, fake_image, reduction='mean')

                face_ID_lossR = face_ID_criterion(source_image_mask, rev_source_image_2_iwt)
                # 更新损失记录器
                Revealed_l2losses.update(lossR.cpu(), fake_image.size(0))
                Revealed_lpiplosses.update(lpips_lossR.cpu(), fake_image.size(0))
                Revealed_wlosses.update(loss_w.cpu(), fake_image.size(0))
                Revealed_idlosses.update(face_ID_lossR.cpu(), fake_image.size(0))

                # 计算总损失并进行反向传播和优化器更新
                # loss = args.alpha[0] * lossH + args.alpha[1] * face_ID_lossH + args.alpha[2] * lpips_lossH
                loss = args.alpha[0] * lossH + args.alpha[1] * face_ID_lossH + args.alpha[2] * lpips_lossH + args.alpha[3] * loss_Bit \
                             + args.alpha[4] * lossR + args.alpha[5] * face_ID_lossR + args.alpha[6] * lpips_lossR

                loss.backward()
                # optimizerR.step()
                optimizerH.step()
                optimizer_decoder.step()
                optimizer_encoder.step()
                Sum_losses.update(loss.cpu(), fake_image.size(0))

            global_step += 1

        # if ((best_loss > Sum_losses.avg.item()) or ((epoch + 1) % 5 == 0)) and (args.train == 1) :
        #     best_loss = Sum_losses.avg.item()
        #     save_network(Source_encoder, '%s/models/Fake_encoder_%s_%.2e.pth' % (args.save_path, epoch,Sum_losses.avg.item()))
        #     save_network(encoder, '%s/models/MBRS_EN_%s_%.2e.pth' % (args.save_path, epoch,Sum_losses.avg.item()))
        #     save_network(decoder, '%s/models/MBRS_DE_%s_%.2e.pth' % (args.save_path, epoch,Sum_losses.avg.item()))

        # # 检查是否达到了进行验证的周期
        # if (epoch + 1) % args.val_epochs == 0:
        #     # 将多个图像拼接成一个输出图像，包括原图、伪造图像、容器图像、变换后的容器图像、变换后的原图、解码后的原图
        #     output_image = torch.cat((transformed_source.cpu(), rev_source_image_3.cpu(), diff_s.cpu(),
        #                               fake_image.cpu(), container_source_image_1_iwt.cpu(), diff_f.cpu()), dim=0)
        #     # 保存输出图像到指定路径
        #     save_image(output_image, '%s/images/output_train_%s.jpg' % (args.save_path, epoch),
        #                normalize=True, nrow=args.batch_size)
        #     # 将面部遮罩图像拼接后保存
        #     output_image = torch.cat((face_mask.cpu(), transformed_face_mask.cpu()), dim=0)
        #     save_image(output_image, '%s/images/output_mask.jpg' % args.save_path, normalize=True, nrow=args.batch_size)

        # # 如果达到了指定的验证周期并且已经超过了开始验证的起始周期
        if (epoch + 1) % args.val_epochs == 0 and epoch > args.start_val_epochs and (args.val == 1):
            # 将模型设置为评估模式
            Source_encoder.train(False)
            Source_encoder.eval()

            # 初始化用于记录验证过程中各种损失的计数器
            Hiding_l2losses = AverageMeter()
            Hiding_lpiplosses = AverageMeter()
            Hiding_flosses = AverageMeter()
            Hiding_idlosses = AverageMeter()
            Revealed_l2losses = AverageMeter()
            Revealed_lpiplosses = AverageMeter()
            Revealed_wlosses = AverageMeter()
            Revealed_idlosses = AverageMeter()
            Sum_losses = AverageMeter()
            psnr1 = 0
            psnr2 = 0
            ssim_s = 0
            ssim_f = 0
            id_sim_f = 0
            id_sim_s = 0
            Error_Rate = 0

        #     # 使用tqdm创建一个验证数据的进度条
            valid_process = tqdm(val_loader,ncols=170)

            # 对于验证过程中的每一批数据
            for idx, (fake_image, source_image, face_mask) in enumerate(valid_process):

                message0 = torch.randint(0, 2, (fake_image.shape[0], 256), dtype=torch.float32).to(device)
                # message = torch.Tensor(np.random.choice([0, 1], (fake_image.shape[0], message_length))).to(device)

                # message = message0.repeat_interleave(8, dim=1)
                message = message0

                # Error_Rate_0 = 100 * decoded_message_error_rate_c(message0, message, message_length,
                #                                                   args.batch_size)
                # 获取message张量的形状
                fake_shape = fake_image.shape

                z_tensor = gauss_noise(fake_shape)
                # 更新验证进度的描述
                if idx > 0:
                    valid_process.set_description(
                        # "Epoch:%d H: l2 %.2e, id %.0f, lp %.2e, R: l2 %.2e, id %.0f, lp %.2e, Sum: %.2e" %
                        # (epoch, Hiding_l2losses.avg.item(), Hiding_idlosses.avg.item(), Hiding_lpiplosses.avg.item(),
                        #  Revealed_l2losses.avg.item(), Revealed_idlosses.avg.item(), Revealed_lpiplosses.avg.item(),
                        #  Sum_losses.avg.item()))
                        'Ep:%d H_m:%.2e, id: %.4f, l:%.2e,  R_m:%.2e, id: %.4f, l:%.2e, R_w:%.2e, PSNR1:%.2f PSNR2:%.2f Sum:%.2e' %
                    (epoch, Hiding_l2losses.avg.item(), Hiding_idlosses.avg.item(), Hiding_lpiplosses.avg.item(),
                     Revealed_l2losses.avg.item(), Revealed_idlosses.avg.item(),Revealed_lpiplosses.avg.item(),Revealed_wlosses.avg.item(),
                     psnr1_avg,psnr2_avg, Sum_losses.avg.item()))


                # 在无梯度模式下进行验证
                with torch.no_grad():
                    # 将图像数据和面具数据移动到GPU，并将它们包装成Variable
                    fake_image = Variable(fake_image.cuda().detach())
                    source_image = Variable(source_image.cuda().detach())
                    face_mask = Variable(face_mask.cuda().detach())

                    source_image_mask = source_image


                    source_image_mask_dwt = dwt(source_image_mask)
                    message_dwt = message
                    fake_image_dwt = dwt(fake_image)
                    source_image_dwt = dwt(source_image)
                    z_tensor_dwt = dwt(z_tensor)

                    source_input = torch.cat([fake_image_dwt, source_image_mask_dwt], dim=1)
                    fake_image_1 = torch.cat([fake_image_dwt, z_tensor_dwt], dim=1)
                    source_image_1 = torch.cat([source_image_dwt, z_tensor_dwt], dim=1)
                    # 使用编码器处理源图像和伪造图像
                    container_source_image = Source_encoder(source_input, rev=False)
                    container_source_image_1 = container_source_image[:, :12, :, :]
                    container_source_image_2 = container_source_image[:, 12:, :, :]

                    container_source_image_1_low = container_source_image_1.narrow(1, 0, 3)
                    fake_image_dwt_low = fake_image_dwt.narrow(1, 0, 3)

                    container_source_image_1_iwt = iwt(container_source_image_1)
                    container_source_image_2_iwt = iwt(container_source_image_2)

                    # 计算隐藏损失
                    lossH = torch.mean(((color.rgb_to_yuv(container_source_image_1_iwt) - color.rgb_to_yuv(fake_image))) ** 2,
                                       axis=[0, 2, 3])  # 计算封装后和伪造图像之间的损失
                    lossH = torch.dot(lossH, yuv_scales)  # 应用YUV缩放

                    # lossH = 2 * F.mse_loss(container_source_image_1_iwt * face_mask, fake_image, reduction='mean')
                    # + 0 * F.mse_loss(container_source_image_1_iwt * (1 - face_mask), fake_image, reduction='mean')
                    lpips_lossH = torch.mean(
                        loss_fn_vgg((fake_image - 0.5) * 2.0, (container_source_image_1_iwt - 0.5) * 2.0))  # 计算LPIPS损失

                    face_ID_lossH = face_ID_criterion(fake_image, container_source_image_1_iwt)  # 计算人脸ID损失
                    # 更新损失计数器
                    Hiding_l2losses.update(lossH.cpu(), fake_image.size(0))
                    Hiding_lpiplosses.update(lpips_lossH.cpu(), fake_image.size(0))
                    Hiding_idlosses.update(face_ID_lossH.cpu(), fake_image.size(0))

                    if args.wm == 1:
                        # container_source_image_en = encoder(fake_image, message)
                        container_source_image_en = encoder(container_source_image_1_iwt, message)

                        container_source_image_1_iwt = quantization(container_source_image_en)
                    else:
                        container_source_image_1_iwt = quantization(container_source_image_1_iwt)

                    if args.noise == 1:
                        container_source_image_1_no = Noise.noise(container_source_image_1_iwt, args.use_noise)
                    else:
                        container_source_image_1_no = container_source_image_1_iwt



                    # transformed_container_source_image_1, transformed_fake, transformed_source, transformed_face_mask = dsl_net(
                    #     container_source_image_1_iwt, fake_image, source_image, face_mask, args, 100,
                    #     Crop_layer, Resize_layer)

                    transformed_container_source_image_1 = container_source_image_1_no
                    transformed_source = source_image_mask
                    transformed_face_mask = face_mask
                    transformed_fake = fake_image


                    transformed_container_source_image_dwt = dwt(transformed_container_source_image_1)
                    transformed_source_dwt = dwt(transformed_source)
                    transformed_face_mask_dwt = dwt(transformed_face_mask)



                    transformed_container_source_image = torch.cat(
                        [transformed_container_source_image_dwt, z_tensor_dwt], dim=1)
                    # 使用解码器获取解码后的源图像
                    rev_source_image = Source_encoder(transformed_container_source_image, rev=True)
                    rev_source_image_1 = rev_source_image[:, :12, :, :]
                    rev_source_image_2 = rev_source_image[:, 12:, :, :]

                    rev_message = decoder(transformed_container_source_image_1)

                    rev_source_image_1_iwt = iwt(rev_source_image_1)
                    rev_source_image_2_iwt = iwt(rev_source_image_2)

                    rev_source_image_3 = rev_source_image_2_iwt * transformed_face_mask + transformed_container_source_image_1 * (
                                1 - transformed_face_mask)
                    # rev_source_image_3 = rev_source_image_2_iwt



                    rev_source_image_3 = quantization(rev_source_image_3)

                    Error_Rate_0 = 100 * (1 - decoded_message_error_rate_c(message0, rev_message.cuda(), message_length,
                                                                      args.batch_size))

                    psnr1_0 = psnr(fake_image[0].cpu(), container_source_image_1_iwt[0].cpu())
                    # psnr1_0 = psnr(container_source_image_1_iwt[0].cpu(), fake_image[0].cpu())

                    psnr2_0 = psnr(transformed_source[0].cpu(), rev_source_image_3[0].cpu())
                    # psnr2_0 = psnr(rev_source_image_3[0].cpu(), transformed_source[0].cpu())

                    psnr1 = psnr1 + psnr1_0
                    psnr2 = psnr2 + psnr2_0
                    psnr1_avg = psnr1 / (idx + 1)
                    psnr2_avg = psnr2 / (idx + 1)
                    ssim_s_0 = ssim(source_image, rev_source_image_3)
                    ssim_f_0 = ssim(fake_image,container_source_image_1_iwt)
                    ssim_s = ssim_s + ssim_s_0
                    ssim_f = ssim_f + ssim_f_0
                    ssim_s_avg = ssim_s / (idx + 1)
                    ssim_f_avg = ssim_f / (idx + 1)
                    Error_Rate = Error_Rate + Error_Rate_0
                    Error_Rate_avg = Error_Rate / (idx + 1)
                    with torch.no_grad():
                        embedding1 = model(source_image)
                        embedding2 = model(rev_source_image_3)
                    id_sim_s_0 = np.dot(embedding1[0].cpu(), embedding2[0].cpu()) / (np.linalg.norm(embedding1[0].cpu()) * np.linalg.norm(embedding2[0].cpu()))
                    id_sim_s = id_sim_s + id_sim_s_0
                    id_sim_s_avg = id_sim_s / (idx + 1)
                    with torch.no_grad():
                        embedding1 = model(fake_image)
                        embedding2 = model(container_source_image_1_iwt)
                    id_sim_f_0 = np.dot(embedding1[0].cpu(), embedding2[0].cpu()) / (np.linalg.norm(embedding1[0].cpu()) * np.linalg.norm(embedding2[0].cpu()))
                    id_sim_f = id_sim_f + id_sim_f_0
                    id_sim_f_avg = id_sim_f / (idx + 1)

                    diff_s = (transformed_source - rev_source_image_3) * 5
                    diff_f = (fake_image - container_source_image_1_iwt) * 5

                    # # 将多个图像拼接成一个输出图像，包括原图、伪造图像、容器图像、变换后的容器图像、变换后的原图、解码后的原图
                    # output_image = torch.cat((transformed_source[0].cpu(),fake_image[0].cpu(),
                    #                            container_source_image_1_iwt[0].cpu(),
                    #                           rev_source_image_3[0].cpu(), diff_f[0].cpu(), diff_s[0].cpu()),
                    #                          dim=1)
                    # # 保存输出图像到指定路径
                    # output_image_np = output_image.numpy()
                    #
                    # output_image = np.clip(output_image_np, 0, 1)
                    # container_save_path = '/home/lab/workspace/works/zqs/project/Face-Recover/show'
                    # name = args.savename
                    # cv2.imwrite(os.path.join(container_save_path,name),
                    #             cv2.cvtColor((output_image.transpose(1, 2, 0) * 255.0).astype(np.uint8),
                    #                          cv2.COLOR_RGB2BGR))


                    if idx % 10 == 0:
                        import matplotlib.pyplot as plt
                        # 假设img_tensor是一个在GPU上的Tensor，形状为[C, H, W]或[N, C, H, W]
                        # 举个例子，我们创建一个假的Tensor用于演示
                        # img_tensor = torch.rand(3, 256, 256, device='cuda')  # 创建一个在GPU上的随机Tensor
                        # 如果Tensor形状是[N, C, H, W]，选择要显示的图片索引，这里我们假设只有一张图片


                        output_1 = torch.cat((transformed_source[0].cpu(), fake_image[0].cpu()), dim=2)
                        output_2 = torch.cat((rev_source_image_3[0].cpu(), container_source_image_1_iwt[0].cpu()),
                                             dim=2)

                        output_3 = torch.cat((diff_s[0].cpu(), diff_f[0].cpu()), dim=2)
                        utput_image_1 = torch.cat((output_1, output_2, output_3), dim=1)
                        # img_tensor = container_source_image_1_iwt[0]  # 选择第一张图片
                        # 将Tensor移动到CPU上
                        img_tensor_cpu = utput_image_1
                        # psnr1 = psnr(fake_image[0].cpu(),container_source_image_1_iwt[0].cpu())
                        # psnr2 = psnr(source_image[0].cpu(),rev_source_image_3[0].cpu())
                        # 将Tensor转换为NumPy数组
                        img_numpy = img_tensor_cpu.detach().numpy()
                        # 如果是3通道图片，调整Tensor形状为[H, W, C]
                        img_numpy = img_numpy.transpose(1, 2, 0)
                        plt.imshow(img_numpy)
                        plt.axis('off')
                        plt.suptitle(f'test: error_rate: {Error_Rate_avg}, \n F&C: {psnr1_avg:.2f},ssim_hide: {ssim_f_avg:.4f},id_sim_hi: {id_sim_f_avg:.4f}\n S&R: {psnr2_avg:.2f},ssim_rev: {ssim_s_avg:.4f},id_sim_rev: {id_sim_s_avg:.4f}'
                                     ,y = 1.002)
                        plt.show()

                    # 计算解码后图像与增强后源图像间的损失
                    # lossR = torch.mean(
                    #     ((color.rgb_to_yuv(rev_source_image) - color.rgb_to_yuv(transformed_source))) ** 2,
                    #     axis=[0, 2, 3])
                    # lossR = torch.dot(lossR, yuv_scales)  # 应用YUV缩放因子
                    lossR = F.mse_loss(rev_source_image_2_iwt, transformed_source, reduction='mean')
                    # 计算LPIPS损失和人脸ID损失
                    lpips_lossR = torch.mean(
                        loss_fn_vgg((source_image_mask - 0.5) * 2.0, (rev_source_image_2_iwt - 0.5) * 2.0))
                    # Error_Rate = decoded_message_error_rate(message0, rev_source_image_2_iwt,message_length, args.batch_size)

                    loss_w = F.mse_loss(rev_source_image_1_iwt, fake_image, reduction='mean')

                    face_ID_lossR = face_ID_criterion(source_image_mask, rev_source_image_2_iwt)
                    # 更新损失记录器
                    Revealed_l2losses.update(lossR.cpu(), fake_image.size(0))
                    Revealed_lpiplosses.update(lpips_lossR.cpu(), fake_image.size(0))
                    Revealed_wlosses.update(loss_w.cpu(), fake_image.size(0))
                    Revealed_idlosses.update(face_ID_lossR.cpu(), fake_image.size(0))

                    # 计算总损失并进行反向传播和优化器更新
                    # loss = args.alpha[0] * lossH + args.alpha[1] * face_ID_lossH + args.alpha[2] * lpips_lossH
                    loss = args.alpha[0] * lossH + args.alpha[1] * face_ID_lossH + args.alpha[2] * lpips_lossH \
                                 + args.alpha[4] * lossR + args.alpha[5] * face_ID_lossR + args.alpha[6] * lpips_lossR
                    # loss.backward()
                    # optimizerR.step()
                    # optimizerH.step()
                    Sum_losses.update(loss.cpu(), fake_image.size(0))


            # # 将多个图像沿着批处理维度拼接起来，以便于可视化比较
            # output_image = torch.cat((transformed_source.cpu(), rev_source_image_3.cpu(), diff_s.cpu(),
            #                          fake_image.cpu(), container_source_image_1_iwt.cpu(), diff_f.cpu()), dim=0)
            # # 保存合并后的图像到指定的文件路径
            # save_image(output_image, '%s/F_en_att_F2F/images/output_test_%s.jpg' % (args.save_path, epoch),
            #            normalize=True, nrow=args.val_batch_size)

            # 创建一个空的DataFrame用于记录本次验证的各项指标
            df_acc = pd.DataFrame()
            # 添加各项指标到DataFrame
            df_acc['epoch'] = [epoch]
            df_acc['Hiding_l2losses'] = [Hiding_l2losses.avg.item()]
            df_acc['Hiding_lpiplosses'] = [Hiding_lpiplosses.avg.item()]
            df_acc['Hiding_wlosses'] = [Hiding_flosses.avg.item()]
            # df_acc['Hiding_idlosses'] = [Hiding_idlosses.avg.item()]
            df_acc['Revealed_l2losses'] = [Revealed_l2losses.avg.item()]
            df_acc['Revealed_lpiplosses'] = [Revealed_lpiplosses.avg.item()]
            df_acc['Revealed_wlosses'] = [Revealed_wlosses.avg.item()]
            # df_acc['Revealed_idlosses'] = [Revealed_idlosses.avg.item()]
            df_acc['Sum_Loss'] = [Sum_losses.avg.item()]

        #     # 根据当前是哪个周期决定是否在CSV文件中包含头部信息
        #     if epoch + 1 != (args.val_epochs + args.start_val_epochs):
        #         df_acc.to_csv('%s/report/validation.csv' % args.save_path, mode='a', index=None, header=None)
        #     else:
        #         df_acc.to_csv('%s/report/validation.csv' % args.save_path, mode='a', index=None)
        #
        #     # 如果当前损失小于之前记录的最佳损失，则更新最佳损失并保存当前的网络模型
        #     if best_loss > Sum_losses.avg.item():
        #         best_loss = Sum_losses.avg.item()
        #         save_network(Source_encoder, '%s/models/Source_encoder.pth' % args.save_path)
        #         # save_network(Fake_encoder, '%s/models/Fake_encoder.pth' % args.save_path)
        #         save_network(Source_encoder, '%s/models/best/Source_encoder_%s.pth' % (args.save_path, epoch))  # 保存编码器模型
        #         # save_network(Fake_encoder, '%s/models/best/Fake_encoder_%s.pth' % (args.save_path, epoch))  # 保存解码器模型
        #
        # # 每100个周期保存一次模型
        # if (epoch + 1) % 100 == 0:
        #     save_network(Source_encoder, '%s/models/train/Source_encoder_%s.pth' % (args.save_path, epoch))  # 保存编码器模型
        #     # save_network(Fake_encoder, '%s/models/train/Fake_encoder_%s.pth' % (args.save_path, epoch))  # 保存解码器模型
        schedulerH.step()  # 更新编码器的学习率
        scheduler_encoder.step()  # 更新编码器的学习率
        scheduler_decoder.step()  # 更新编码器的学习率
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

