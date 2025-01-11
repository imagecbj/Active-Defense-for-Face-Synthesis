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
from invertible_net import Model_att
from torchvision.utils import save_image

from transforms import build_transforms
from utils import AverageMeter, psnr, gauss_noise, ssim,quantization
from utils import decoded_message_error_rate
from utils import message_expand,DWT,IWT, low_frequency_loss
from ff_df import ff_df_Dataloader
import torch.nn.functional as F

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

    parser.add_argument('--gpu_id', type=int, default=0)

    parser.add_argument('--rnd_bri_ramp', type=int, default=5000)
    parser.add_argument('--rnd_hue_ramp', type=int, default=5000)
    parser.add_argument('--jpeg_quality_ramp', type=float, default=5000)
    parser.add_argument('--rnd_noise_ramp', type=int, default=5000)
    parser.add_argument('--contrast_ramp', type=int, default=5000)
    parser.add_argument('--rnd_sat_ramp', type=int, default=5000)
    parser.add_argument('--rnd_crop_ramp', type=int, default=5000)
    parser.add_argument('--rnd_resize_ramp', type=int, default=5000)
    parser.add_argument('--rnd_bri', type=float, default=.1)
    parser.add_argument('--rnd_hue', type=float, default=.05)
    parser.add_argument('--jpeg_quality', type=float, default=50)
    parser.add_argument('--rnd_noise', type=float, default=.02)
    parser.add_argument('--contrast_low', type=float, default=.8)
    parser.add_argument('--contrast_high', type=float, default=1.2)
    parser.add_argument('--rnd_sat', type=float, default=0.5)
    parser.add_argument('--blur_prob', type=float, default=0.1)
    parser.add_argument('--no_jpeg', type=bool, default=False)
    parser.add_argument('--is_cuda', type=bool, default=False)
    parser.add_argument('--rnd_crop', type=float, default=0.1)
    parser.add_argument('--rnd_resize', type=float, default=0)  

    parser.add_argument('--message_length', type=int, default=64)
    parser.add_argument('--message_range', type=float, default=1)

    parser.add_argument('--y_scale', type=float, default=1.0)
    parser.add_argument('--u_scale', type=float, default=10.0)
    parser.add_argument('--v_scale', type=float, default=10.0)

    parser.add_argument('--train', type=int, default=1)
    parser.add_argument('--test', type=int, default=0)
    parser.add_argument('--noise', type=int, default=1)
    parser.add_argument('--choose_data', type=int, default=1)

    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--val_epochs', type=int, default=20)
    parser.add_argument('--start_val_epochs', type=int, default=50)
    parser.add_argument('--adjust_lr_epochs', type=int, default=50)
    parser.add_argument('--base_lr', type=float, default=0.00001)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--resolution', type=int, default=256)
    parser.add_argument('--val_batch_size', type=int, default=4)
    parser.add_argument('--alpha', type=list, default=[1, 0.85, 1,0, 1, 0.85, 1 , 0])
    parser.add_argument('--root_path', type=str, default='/home/lab/workspace/works/zqs/datasets/FF++')
    # parser.add_argument('--root_path', type=str, default='/home/lab/workspace/works/zqs/datasets/Celeb_DF')
    parser.add_argument('--save_path', type=str, default='./save_results_NM/F_en_att_DF')

    args = parser.parse_args()
    return args
def main():
    args = parse_args()

    transform_train, transform_test = build_transforms(args.resolution, args.resolution,
                                                       max_pixel_value=255.0, norm_mean=[0, 0, 0],
                                                       norm_std=[1.0, 1.0, 1.0])
    yuv_scales = torch.Tensor([args.y_scale, args.u_scale, args.v_scale]).cuda()
    # with open('./save_txt/CDF/train_cdf_fake.txt', 'r') as f:
    #     fake_train_videos = f.readlines()
    #     fake_train_videos = [i.strip() for i in fake_train_videos]
    # with open('./save_txt/CDF/train_cdf_real.txt', 'r') as f:
    #     source_train_videos = f.readlines()
    #     source_train_videos = [i.strip() for i in source_train_videos]
    #
    #
    # with open('./save_txt/CDF/test_cdf_fake.txt', 'r') as f:
    #     fake_val_videos = f.readlines()
    #     fake_val_videos = [i.strip() for i in fake_val_videos]
    #
    # with open('./save_txt/CDF/test_cdf_real.txt', 'r') as f:
    #     source_val_videos = f.readlines()
    #     source_val_videos = [i.strip() for i in source_val_videos]

    with open('./save_txt/Deepfakes/c23/train_df_fake_c23.txt', 'r') as f:
        fake_train_videos = f.readlines()
        fake_train_videos = [i.strip() for i in fake_train_videos]
    #
    with open('./save_txt/Deepfakes/c23/train_df_real_c23.txt', 'r') as f:
        source_train_videos = f.readlines()
        source_train_videos = [i.strip() for i in source_train_videos]
    #
    #
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
    print('Use Train videos Number: %d' % len(train_dataset))

    dwt = DWT()
    iwt = IWT()

    # 初始化编码器模型，并将其移至GPU上
    Source_encoder = Model_att().cuda()
    file_path = '/home/lab/workspace/works/zqs/project/Face-Recover/save_results_2/F_en_att_DF/models/Fake_encoder11.pth'
    if os.path.exists(file_path):
        state_dicts = torch.load(file_path)
        print('Cannot load optimizer for some reason or other')
        Source_encoder.load_state_dict(state_dicts)
        try:
            optim.load_state_dict(state_dicts['opt'])
        except:
            print('Cannot load optimizer for some reason or other')
    best_loss = 100

    # 创建裁剪层，用于随机裁剪图像
    Crop_layer = Crop([1.0 - args.rnd_crop, 1.0], [1.0 - args.rnd_crop, 1.0])
    # 创建调整大小层，用于随机调整图像大小
    Resize_layer = Resize(1.0 - args.rnd_resize, 1.0 + args.rnd_resize)

    # 为编码器模型创建Adam优化器
    optimizerH = optim.Adam(filter(lambda p: p.requires_grad,
                                   Source_encoder.parameters()), lr=args.base_lr)


    # 为编码器设置余弦退火学习率调度器
    schedulerH = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerH, T_max=args.adjust_lr_epochs)

    # 初始化LPIPS损失函数，用于评估图像间的感知差异
    loss_fn_vgg = lpips.LPIPS(net='vgg').eval().cuda()
    # 初始化Arcface损失函数，用于人脸识别任务
    face_ID_criterion = Arcface_loss().cuda()
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
        Hiding_flosses = AverageMeter()
        Hiding_idlosses = AverageMeter()
        Revealed_l2losses = AverageMeter()
        Revealed_lpiplosses = AverageMeter()
        Revealed_idlosses = AverageMeter()
        Revealed_wlosses = AverageMeter()
        Sum_losses = AverageMeter()
        training_process = tqdm(train_loader,ncols=170)  # 使用tqdm创建一个进度条
        psnr1 = 0
        psnr2 = 0
        ssim_s = 0
        id_sim = 0
        # 遍历训练数据加载器中的数据
        if args.train == 1:
            for idx, (fake_image, source_image, face_mask) in enumerate(training_process):

                message0 = torch.Tensor(np.random.choice([-message_range, message_range],
                                                        (fake_image.shape[0], message_length )  )).to(device)

                message = message_expand(message0, message_length)

                # ww = decoded_message_error_rate(message0,message,message_length,args.batch_size)
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
                        'Ep:%d H_m:%.2e, id: %.4f, l:%.2e, H_w:%.2e, R_m:%.2e, id: %.4f, l:%.2e, R_w:%.2e, PSNR1:%.2f PSNR2:%.2f Sum:%.2e' %
                        (epoch, Hiding_l2losses.avg.item(), Hiding_idlosses.avg.item(), Hiding_lpiplosses.avg.item(),Hiding_flosses.avg.item(),
                         Revealed_l2losses.avg.item(), Revealed_idlosses.avg.item(),Revealed_lpiplosses.avg.item(),Revealed_wlosses.avg.item(),
                         psnr1_avg,psnr2_avg, Sum_losses.avg.item()))
                optimizerH.zero_grad()  # 清零编码器的梯度
                # optimizerR.zero_grad()  # 清零解码器的梯度


                if 1:
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
                    lossH = torch.mean(((color.rgb_to_yuv(container_source_image_1_iwt ) - color.rgb_to_yuv(fake_image))) ** 2,
                                       axis=[0, 2, 3])  # 计算封装后和伪造图像之间的损失
                    lossH = torch.dot(lossH, yuv_scales)  # 应用YUV缩放

                    lpips_lossH = torch.mean(
                        loss_fn_vgg((fake_image - 0.5) * 2.0, (container_source_image_1_iwt - 0.5) * 2.0))  # 计算LPIPS损失
                    lossH_f = F.mse_loss(container_source_image_2, z_tensor_dwt, reduction='mean')
                    face_ID_lossH = face_ID_criterion(container_source_image_1_iwt, fake_image)  # 计算人脸ID损失
                    # 更新损失计数器
                    Hiding_l2losses.update(lossH.cpu(), fake_image.size(0))
                    Hiding_lpiplosses.update(lpips_lossH.cpu(), fake_image.size(0))
                    Hiding_flosses.update(lossH_f.cpu(), fake_image.size(0))
                    Hiding_idlosses.update(face_ID_lossH.cpu(), fake_image.size(0))
                    # 数据增强处理，包括裁剪和调整大小

                    if (epoch % 15 == 0) and (epoch > 400):
                    # if 1:
                        transformed_container_source_image_1 = container_source_image_1_iwt
                        transformed_source = source_image_mask
                        transformed_face_mask = face_mask
                    else:
                        transformed_container_source_image_1, transformed_fake, transformed_source, transformed_face_mask = dsl_net(
                        container_source_image_1_iwt, fake_image, source_image, face_mask, args, global_step, Crop_layer, Resize_layer)


                    transformed_container_source_image_dwt = dwt(transformed_container_source_image_1)
                    transformed_source_dwt = dwt(transformed_source)
                    transformed_face_mask_dwt = dwt(transformed_face_mask)

                    transformed_container_source_image = torch.cat([transformed_container_source_image_dwt, zero_tensor_dwt], dim=1)
                    # 使用解码器获取解码后的源图像
                    rev_source_image = Source_encoder(transformed_container_source_image, rev=True)
                    rev_source_image_1 = rev_source_image[:, :12, :, :]
                    rev_source_image_2 = rev_source_image[:, 12:, :, :]


                    rev_source_image_1_iwt = iwt(rev_source_image_1)
                    rev_source_image_2_iwt = iwt(rev_source_image_2)


                    rev_source_image_3 = rev_source_image_2_iwt

                    psnr1_0 = psnr(fake_image[0].cpu(), container_source_image_1_iwt[0].cpu())
                    psnr2_0 = psnr(transformed_source[0].cpu(), rev_source_image_3[0].cpu())
                    psnr1 = psnr1 + psnr1_0
                    psnr2 = psnr2 + psnr2_0
                    psnr1_avg = psnr1/(idx+1)
                    psnr2_avg = psnr2/(idx+1)
                    ssim_s_0 = ssim(transformed_source, rev_source_image_3)
                    ssim_s = ssim_s + ssim_s_0
                    ssim_s_avg = ssim_s / (idx + 1)
                    with torch.no_grad():
                        embedding1 = model(transformed_source)
                        embedding2 = model(rev_source_image_3)
                    id_sim_0 = np.dot(embedding1[0].cpu(), embedding2[0].cpu()) / (
                                np.linalg.norm(embedding1[0].cpu()) * np.linalg.norm(embedding2[0].cpu()))
                    id_sim = id_sim + id_sim_0
                    id_sim_avg = id_sim / (idx + 1)

                    if (idx % 40 ==0 )& (epoch % 20 == 0)  :

                        import matplotlib.pyplot as plt

                        output_1 = torch.cat((transformed_source[0].cpu(), fake_image[0].cpu()), dim=2)
                        output_2 = torch.cat((rev_source_image_3[0].cpu(), container_source_image_1_iwt[0].cpu()), dim=2)
                        diff_s = (transformed_source - rev_source_image_3) * 5
                        diff_f = (fake_image - container_source_image_1_iwt) * 5
                        output_3 = torch.cat((diff_s[0].cpu(), diff_f[0].cpu()), dim=2)
                        utput_image_1 = torch.cat((output_1, output_2, output_3), dim=1)
                        # img_tensor = container_source_image_1_iwt[0]  # 选择第一张图片
                        # 将Tensor移动到CPU上
                        img_tensor_cpu = utput_image_1
                        # 将Tensor转换为NumPy数组
                        img_numpy = img_tensor_cpu.detach().numpy()
                        # 如果是3通道图片，调整Tensor形状为[H, W, C]
                        img_numpy = img_numpy.transpose(1, 2, 0)
                        plt.imshow(img_numpy)
                        plt.axis('off')
                        plt.suptitle(f'ssim: {ssim_s_avg:.2f}, id_sim: {id_sim_avg:.2f}\n S&R: {psnr2_avg:.2f},F&C: {psnr1_avg:.2f}')
                        plt.show()


                    # 计算解码后图像与增强后源图像间的损失
                    lossR = torch.mean(
                        ((color.rgb_to_yuv(transformed_source) - color.rgb_to_yuv(rev_source_image_2_iwt))) ** 2,
                        axis=[0, 2, 3])
                    lossR = torch.dot(lossR, yuv_scales)  # 应用YUV缩放因子
                    # lossR = F.mse_loss(rev_source_image_2_iwt, transformed_source, reduction='mean')
                    # 计算LPIPS损失和人脸ID损失
                    lpips_lossR = torch.mean(
                        loss_fn_vgg((source_image_mask - 0.5) * 2.0, (rev_source_image_2_iwt - 0.5) * 2.0))

                    loss_w = F.mse_loss(rev_source_image_1_iwt, fake_image, reduction='mean')

                    face_ID_lossR = face_ID_criterion(transformed_source, rev_source_image_1_iwt)
                    # 更新损失记录器
                    Revealed_l2losses.update(lossR.cpu(), fake_image.size(0))
                    Revealed_lpiplosses.update(lpips_lossR.cpu(), fake_image.size(0))
                    Revealed_wlosses.update(loss_w.cpu(), fake_image.size(0))
                    Revealed_idlosses.update(face_ID_lossR.cpu(), fake_image.size(0))

                    # 计算总损失并进行反向传播和优化器更新
                    # loss = args.alpha[0] * lossH + args.alpha[1] * face_ID_lossH + args.alpha[2] * lpips_lossH
                    loss = args.alpha[0] * lossH + args.alpha[1] * face_ID_lossH + args.alpha[2] * lpips_lossH \
                     + args.alpha[4] * lossR + args.alpha[5] * face_ID_lossR + args.alpha[6] * lpips_lossR
                    loss.backward()
                    # optimizerR.step()
                    optimizerH.step()
                    schedulerH.step()  # 更新编码器的学习率

                    # 更新总损失和全局步数
                Sum_losses.update(loss.cpu(), fake_image.size(0))

                global_step += 1

            save_network(Source_encoder, '%s/models/Fake_encoder_%s.pth' % (args.save_path, epoch))

            # 检查是否达到了进行验证的周期
            if (epoch + 1) % args.val_epochs == 0:
                # 将多个图像拼接成一个输出图像，包括原图、伪造图像、容器图像、变换后的容器图像、变换后的原图、解码后的原图
                output_image = torch.cat((transformed_source.cpu(), rev_source_image_3.cpu(), diff_s.cpu(),
                                          fake_image.cpu(), container_source_image_1_iwt.cpu(), diff_f.cpu()), dim=0)
                # 保存输出图像到指定路径
                save_image(output_image, '%s/images/output_train_%s.jpg' % (args.save_path, epoch),
                           normalize=True, nrow=args.batch_size)
                # 将面部遮罩图像拼接后保存
                output_image = torch.cat((face_mask.cpu(), transformed_face_mask.cpu()), dim=0)
                save_image(output_image, '%s/images/output_mask.jpg' % args.save_path, normalize=True, nrow=args.batch_size)

        # # 如果达到了指定的验证周期并且已经超过了开始验证的起始周期
        if (epoch + 1) % args.val_epochs == 0 and epoch > args.start_val_epochs and args.test == 1:
            # 将模型设置为评估模式
            Source_encoder.train(False)
            # Fake_encoder.train(False)
            Source_encoder.eval()
            # Fake_encoder.eval()

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
            id_sim = 0

        #     # 使用tqdm创建一个验证数据的进度条
            valid_process = tqdm(val_loader,ncols=170)

            # 对于验证过程中的每一批数据
            for idx, (fake_image, source_image, face_mask) in enumerate(valid_process):

                message0 = torch.Tensor(np.random.choice([-message_range, message_range],
                                                         (fake_image.shape[0], message_length))).to(device)

                message = message_expand(message0, message_length)
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
                        'Ep:%d H_m:%.2e, id: %.4f, l:%.2e, H_w:%.2e, R_m:%.2e, id: %.4f, l:%.2e, R_w:%.2e, PSNR1:%.2f PSNR2:%.2f Sum:%.2e' %
                    (epoch, Hiding_l2losses.avg.item(), Hiding_idlosses.avg.item(), Hiding_lpiplosses.avg.item(),Hiding_flosses.avg.item(),
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

                    lpips_lossH = torch.mean(
                        loss_fn_vgg((fake_image - 0.5) * 2.0, (container_source_image_1_iwt - 0.5) * 2.0))  # 计算LPIPS损失
                    lossH_f = F.mse_loss(container_source_image_2, z_tensor_dwt, reduction='mean')
                    face_ID_lossH = face_ID_criterion(fake_image, container_source_image_1_iwt)  # 计算人脸ID损失
                    # 更新损失计数器
                    Hiding_l2losses.update(lossH.cpu(), fake_image.size(0))
                    Hiding_lpiplosses.update(lpips_lossH.cpu(), fake_image.size(0))
                    Hiding_flosses.update(lossH_f.cpu(), fake_image.size(0))
                    Hiding_idlosses.update(face_ID_lossH.cpu(), fake_image.size(0))


                    container_source_image_1_iwt = quantization(container_source_image_1_iwt)

                    # transformed_container_source_image_1, transformed_fake, transformed_source, transformed_face_mask = dsl_net(
                    #     container_source_image_1_iwt, fake_image, source_image, face_mask, args, 0,
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

                    rev_source_image_1_iwt = iwt(rev_source_image_1)
                    rev_source_image_2_iwt = iwt(rev_source_image_2)

                    rev_source_image_3 = rev_source_image_2_iwt



                    rev_source_image_3 = quantization(rev_source_image_3)

                    psnr1_0 = psnr(fake_image[0].cpu(), container_source_image_1_iwt[0].cpu())
                    # psnr1_0 = psnr(container_source_image_1_iwt[0].cpu(), fake_image[0].cpu())

                    psnr2_0 = psnr(transformed_source[0].cpu(), rev_source_image_3[0].cpu())
                    # psnr2_0 = psnr(rev_source_image_3[0].cpu(), transformed_source[0].cpu())

                    psnr1 = psnr1 + psnr1_0
                    psnr2 = psnr2 + psnr2_0
                    psnr1_avg = psnr1 / (idx + 1)
                    psnr2_avg = psnr2 / (idx + 1)
                    ssim_s_0 = ssim(source_image, rev_source_image_3)
                    ssim_s = ssim_s + ssim_s_0
                    ssim_s_avg = ssim_s / (idx + 1)
                    with torch.no_grad():
                        embedding1 = model(source_image)
                        embedding2 = model(rev_source_image_3)
                    id_sim_0 = np.dot(embedding1[0].cpu(), embedding2[0].cpu()) / (np.linalg.norm(embedding1[0].cpu()) * np.linalg.norm(embedding2[0].cpu()))
                    id_sim = id_sim + id_sim_0
                    id_sim_avg = id_sim / (idx + 1)


                    if idx % 10 == 0:
                        import matplotlib.pyplot as plt

                        output_1 = torch.cat((transformed_source[0].cpu(), fake_image[0].cpu()), dim=2)
                        output_2 = torch.cat((rev_source_image_3[0].cpu(), iwt(container_source_image_1)[0].cpu()),
                                             dim=2)
                        diff_s = (transformed_source - rev_source_image_3) * 5
                        diff_f = (fake_image - container_source_image_1_iwt) * 5
                        output_3 = torch.cat((diff_s[0].cpu(), diff_f[0].cpu()), dim=2)
                        utput_image_1 = torch.cat((output_1, output_2, output_3), dim=1)
                        # img_tensor = container_source_image_1_iwt[0]  # 选择第一张图片
                        # 将Tensor移动到CPU上
                        img_tensor_cpu = utput_image_1

                        # 将Tensor转换为NumPy数组
                        img_numpy = img_tensor_cpu.detach().numpy()
                        # 如果是3通道图片，调整Tensor形状为[H, W, C]
                        img_numpy = img_numpy.transpose(1, 2, 0)
                        plt.imshow(img_numpy)
                        plt.axis('off')
                        plt.suptitle(f' test:ssim: {ssim_s_avg:.4f},id_sim: {id_sim_avg:.4f},\n S&R: {psnr2_avg:.2f},F&C: {psnr1_avg:.2f}')
                        plt.show()

                    # 计算解码后图像与增强后源图像间的损失
                    # lossR = torch.mean(
                    #     ((color.rgb_to_yuv(rev_source_image_2_iwt) - color.rgb_to_yuv(transformed_source))) ** 2,
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
                    loss = args.alpha[0] * lossH + args.alpha[1] * lpips_lossH + args.alpha[2] * lossH_f + args.alpha[
                        3] * lossR + args.alpha[
                               4] * lpips_lossR + args.alpha[5] * loss_w
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
        # schedulerH.step()  # 更新编码器的学习率
        # schedulerR.step()  # 更新解码器的学习率
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

