import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from random import random,randint
import random as ra
import math
import cv2
# import config
from kornia.geometry.transform.imgwarp import get_perspective_transform
from kornia.geometry.transform.imgwarp import warp_perspective
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
def noise(x,use_noise):
    crop = Crop()
    resize = Resize(0.8,0.8)
    scale_noise = Scale_noise()
    scale_decode_train = Scale_decode_train()
    scale_decode_eval = Scale_decode_eval()
    gau_noise = GaussianNoise()
    screenshooting_noise = ScreenShooting()
    dropout_noise = Dropout([0.8, 0.8])
    salt_pepper_noise = Salt_pepper(0.05)

    # choice = randint(0, 5)
    choice = torch.randint(0, 6, (1,)).item()
    # use_noise = use_noise
    choice = 5
    if use_noise:
        if choice == 0:
            return crop(x)
        if choice == 1:
            return scale_noise(x)
        if choice == 2:
            return gaussian_blur(x)
        if choice == 3:
            return gau_noise(x)
        if choice == 4:
            return adjust_brightness(x,0.8)
        if choice == 5:
            return jpeg_test(x,75)
        if choice == 6:
            return adjust_contrast(x,1.2)

import torch

def adjust_contrast(image, contrast_factor):

    mean = torch.mean(image, dim=(1, 2), keepdim=True)  # 保持通道维度

    # 调整对比度
    adjusted_image = (image - mean) * contrast_factor + mean

    # 确保输出的值仍然在 [0, 1] 范围内
    adjusted_image = torch.clamp(adjusted_image, 0, 1)

    return adjusted_image

# 示例用法
# 假设 image 是一个形状为 [3, 256, 256] 的图像张量，值范围在 [0, 1]
image = torch.rand(3, 256, 256)

# 调整对比度因子，例如 1.5 表示增加对比度，0.8 表示减少对比度
contrast_factor = 1.5
adjusted_image = adjust_contrast(image, contrast_factor)

print(adjusted_image.shape)  # 输出：torch.Size([3, 256, 256])


def jpeg_test(image_tensor, quality):
    """
    对 [0, 1] 范围的 CUDA 张量批次进行 JPEG 压缩。
    输入：
        image_tensor: 含水印的图像 (PyTorch CUDA 张量)，形状为 [b, c, h, w]，值范围为 [0, 1]。
        quality: JPEG 压缩质量 (1-100)。
    输出：
        压缩后的图像 (PyTorch 张量)，形状为 [b, c, h, w]。
    """
    # 检查是否是 CUDA 张量，如果是则转移到 CPU
    if image_tensor.is_cuda:
        image_tensor = image_tensor.cpu()

    # 将张量值从 [0, 1] 范围映射到 [0, 255] 并转换为 uint8
    image_tensor = (image_tensor * 255).clamp(0, 255).to(dtype=torch.uint8)

    # 获取批次大小
    batch_size, channels, height, width = image_tensor.shape
    compressed_images = []

    for i in range(batch_size):
        # 取出批次中的单个图像，形状为 [c, h, w]
        single_image = image_tensor[i]

        # 将单个图像从 [c, h, w] 转换为 [h, w, c]，并转换为 NumPy 数组
        image_np = single_image.permute(1, 2, 0).numpy()

        # 使用 imencode 进行 JPEG 压缩并将其保存在内存中
        is_success, buffer = cv2.imencode('.jpg', image_np, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not is_success:
            raise RuntimeError("图像压缩失败")

        # 将压缩后的图像从缓冲区中加载回来
        compressed_image_np = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

        # 将 NumPy 数组转换回 PyTorch 张量，并将通道顺序从 HWC 转为 CHW
        compressed_image_tensor = torch.from_numpy(compressed_image_np).permute(2, 0, 1)

        # 确保形状一致，避免颜色通道数不匹配
        if channels == 1:
            compressed_image_tensor = compressed_image_tensor.mean(dim=0, keepdim=True)

        compressed_images.append(compressed_image_tensor)

    # 将所有压缩后的图像堆叠回批次张量
    compressed_batch_tensor = torch.stack(compressed_images)

    # 将值重新映射到 [0, 1]
    compressed_batch_tensor = compressed_batch_tensor.to(dtype=torch.float32) / 255.0

    return compressed_batch_tensor.cuda()
def jpeg_compression_train(images):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    jp = JpegCompression(device)
    return jp(images)

class Dropout(nn.Module):
    """
    Drops random pixels from the noised image and substitues them with the pixels from the cover image
    """
    def __init__(self, keep_ratio_range):
        super(Dropout, self).__init__()
        self.keep_min = keep_ratio_range[0]
        self.keep_max = keep_ratio_range[1]

    def forward(self, images):
        mask_percent = np.random.uniform(self.keep_min, self.keep_max)
        mask = np.random.choice([0.0, 1.0], images.shape[2:], p=[1 - mask_percent, mask_percent])
        mask_tensor = torch.tensor(mask, device=images.device, dtype=torch.float)
        mask_tensor = mask_tensor.expand_as(images)
        noised_images = images * mask_tensor
        return noised_images

def random_float(min_, max_):
    return ((np.random.rand() * (max_ - min_) + min_) * 100 // 4 * 4) / 100

class Scale_noise(nn.Module):
    def __init__(self, min_pct=0.6, max_pct=0.6, interpolation_method='bilinear'):
        super(Scale_noise, self).__init__()
        self.min_pct = min_pct
        self.max_pct = max_pct
        self.interpolation_method = interpolation_method

    def forward(self, images):
        # [B, C, H, W]
        batch_size, _, height, width = images.size()
        resize_ratio_H = random_float(self.min_pct, self.max_pct)
        resize_ratio_W = random_float(self.min_pct, self.max_pct)
        height, width = (resize_ratio_H * height // 4) * 4, (resize_ratio_W * width // 4) * 4
        list_images = []
        for i in range(batch_size):
            list_images.append(F.interpolate(
                images[[i], :, :, :],
                size=(int(height), int(width)),
                mode=self.interpolation_method,
                align_corners=True).squeeze(0)
                               )
        scaled_images = torch.stack(list_images, dim=0)
        scaled_images = F.interpolate(scaled_images, size=(256, 256), mode='bilinear', align_corners=False)
        return scaled_images


class Scale_decode_train(nn.Module):
    def __init__(self, interpolation_method='bilinear'):
        super(Scale_decode_train, self).__init__()
        self.interpolation_method = interpolation_method

    def forward(self, images):
        batch_size, _, height, width = images.size()
        list_images = []
        for i in range(batch_size):
            list_images.append(F.interpolate(
                images[[i], :, :, :],
                size=(512, 512),
                mode=self.interpolation_method,
                align_corners=True).squeeze(0)
                               )
        scaled_images = torch.stack(list_images, dim=0)
        return scaled_images


def adjust_brightness(image, brightness_factor):


    # 调整亮度
    adjusted_image = image * brightness_factor

    # 确保输出的值仍然在 [0, 1] 范围内
    adjusted_image = torch.clamp(adjusted_image, 0, 1)

    return adjusted_image
class Scale_decode_eval(nn.Module):
    def __init__(self, interpolation_method='bilinear'):
        super(Scale_decode_eval, self).__init__()
        self.interpolation_method = interpolation_method

    def forward(self, images):
        batch_size, _, height, width = images.size()
        list_images = []
        for i in range(batch_size):
            list_images.append(F.interpolate(
                images[[i], :, :, :],
                size=(768, 768),
                mode=self.interpolation_method,
                align_corners=True).squeeze(0)
                               )
        scaled_images = torch.stack(list_images, dim=0)
        return scaled_images


class Scale_decode_test(nn.Module):
    def __init__(self, interpolation_method='bilinear'):
        super(Scale_decode_test, self).__init__()
        self.interpolation_method = interpolation_method

    def forward(self, images):
        batch_size, _, height, width = images.size()
        list_images = []
        for i in range(batch_size):
            list_images.append(F.interpolate(
                images[[i], :, :, :],
                size=(384, 384),
                mode=self.interpolation_method,
                align_corners=True).squeeze(0)
                               )
        scaled_images = torch.stack(list_images, dim=0)
        return scaled_images


class Scale_decode_big(nn.Module):
    def __init__(self, interpolation_method='bilinear'):
        super(Scale_decode_big, self).__init__()
        self.interpolation_method = interpolation_method

    def forward(self, images):
        batch_size, _, height, width = images.size()
        list_images = []
        for i in range(batch_size):
            list_images.append(F.interpolate(
                images[[i], :, :, :],
                size=(1024, 1024),
                mode=self.interpolation_method,
                align_corners=True).squeeze(0)
                               )
        scaled_images = torch.stack(list_images, dim=0)
        return scaled_images


class Crop(nn.Module):
    def __init__(self, min_pct=0.8, max_pct=0.8):
        super(Crop, self).__init__()
        self.min_pct = min_pct
        self.max_pct = max_pct

    def _pct(self):
        return self.min_pct + random() * (self.max_pct - self.min_pct)

    def forward(self, images):
        _, _, height, width = images.size()
        r = self._pct()

        dx = int(r * width)
        dy = int(r * height)
        # dx = int(0.4 * width)
        # dy = int(0.4 * height)

        dx, dy = (dx // 4) * 4, (dy // 4) * 4
        x = randint(0, width - dx - 1)
        y = randint(0, height - dy - 1)

        crop_mask = images.clone()
        crop_mask[:, :, :, :] = 0.0
        crop_mask[:, :, y:y + dy, x:x + dx] = 1.0
        return images * crop_mask


# 高斯模糊
def get_gaussian_kernel(kernel_size=5, sigma=0.5, channels=3):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2 * variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, groups=channels,
                                bias=False, padding=kernel_size // 2)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter


def gaussian_blur(images):
    # 获取 CUDA 上的高斯卷积核
    channels = images.shape[1]
    kernel = get_gaussian_kernel(channels=channels).cuda()
    # 应用高斯卷积核对输入图像进行模糊处理
    return kernel(images)

class Resize(nn.Module):
    """
    Resize the image. The target size is original size * resize_ratio
    """
    def __init__(self, resize_ratio_min, resize_ratio_max, interpolation_method='bilinear'):
        super(Resize, self).__init__()
        self.resize_ratio_min = resize_ratio_min
        self.resize_ratio_max = resize_ratio_min
        self.interpolation_method = interpolation_method


    def forward(self, container, fake_image, secret_image, face_mask):

        resize_ratio = random_float(self.resize_ratio_min, self.resize_ratio_max)
        container_resize = F.interpolate(
                                    container,
                                    scale_factor=(resize_ratio, resize_ratio),
                                    mode=self.interpolation_method)
        fake_resize = F.interpolate(
                                    fake_image,
                                    scale_factor=(resize_ratio, resize_ratio),
                                    mode=self.interpolation_method)
        secret_resize = F.interpolate(
                                    secret_image,
                                    scale_factor=(resize_ratio, resize_ratio),
                                    mode=self.interpolation_method)
        face_mask_resize = F.interpolate(
                                    face_mask,
                                    scale_factor=(resize_ratio, resize_ratio),
                                    mode=self.interpolation_method)
        return container_resize, fake_resize, secret_resize, face_mask_resize
# 高斯噪声
class GaussianNoise(nn.Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """

    def __init__(self, sigma=0.04, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.noise = torch.tensor(0.).cuda()

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = self.noise.repeat(*x.size()).normal_() * scale
            x = x + sampled_noise
        return x


# 透视扭曲 [B, C, H, W]
def perspective(image, device, d=8):
    # the source points are the region to crop corners
    c = image.shape[0]
    h = image.shape[2]
    w = image.shape[3]  # destination size
    image_size = h
    points_src = torch.ones(c, 4, 2)
    points_dst = torch.ones(c, 4, 2)
    for i in range(c):
        points_src[i, :, :] = torch.tensor([[
            [0., 0.], [w - 1., 0.], [w - 1., h - 1.], [0., h - 1.],
        ]])

        # the destination points are the image vertexes
        # d=8
        tl_x = ra.uniform(-d, d)  # Top left corner, top
        tl_y = ra.uniform(-d, d)  # Top left corner, left

        bl_x = ra.uniform(-d, d)  # Bot left corner, bot
        bl_y = ra.uniform(-d, d)  # Bot left corner, left

        tr_x = ra.uniform(-d, d)  # Top right corner, top
        tr_y = ra.uniform(-d, d)  # Top right corner, right

        br_x = ra.uniform(-d, d)  # Bot right corner, bot
        br_y = ra.uniform(-d, d)  # Bot right corner, right

        points_dst[i, :, :] = torch.tensor([[
            [tl_x, tl_y],
            [tr_x + image_size, tr_y],
            [br_x + image_size, br_y + image_size],
            [bl_x, bl_y + image_size],
        ]])
        # compute perspective transform
    # M: torch.tensor = kornia.get_perspective_transform(points_src, points_dst).to(device)
    M: torch.tensor = get_perspective_transform(points_src, points_dst).to(device)

    # warp the original image by the found transform
    # data_warp: torch.tensor = kornia.warp_perspective(image.float(), M, dsize=(h, w)).to(device)
    data_warp: torch.tensor = warp_perspective(image.float(), M, dsize=(h, w)).to(device)

    return data_warp


# 光照扭曲
def Light_Distortion(c, embed_image):
    mask = np.zeros((embed_image.shape))
    mask_2d = np.zeros((embed_image.shape[2], embed_image.shape[3]))

    # 最小的照明变化比例，在 [0.7, 0.9] 均匀采样
    a = 0.7 + np.random.rand(1) * 0.2
    # 最大的照明变化比例，在 [1.1, 1.3] 均匀采样
    b = 1.1 + np.random.rand(1) * 0.2

    # 线光源
    if c == 0:
        direction = np.random.randint(1, 5)
        for i in range(embed_image.shape[2]):
            mask_2d[i, :] = -((b - a) / (mask.shape[2] - 1)) * (i - mask.shape[3]) + a
        if direction == 1:
            O = mask_2d
        elif direction == 2:
            O = np.rot90(mask_2d, 1)  # 有问题
        elif direction == 3:
            O = np.rot90(mask_2d, 1)
        elif direction == 4:
            O = np.rot90(mask_2d, 1)
        # 线光源分布矩阵
        for batch in range(embed_image.shape[0]):
            for channel in range(embed_image.shape[1]):
                mask[batch, channel, :, :] = mask_2d
    # 点光源
    else:
        # 模拟的点光源坐标
        x = np.random.randint(0, mask.shape[2])
        y = np.random.randint(0, mask.shape[3])

        # 点光源坐标到图像四个角的最大距离
        max_len = np.max([np.sqrt(x ** 2 + y ** 2), np.sqrt((x - 255) ** 2 + y ** 2), np.sqrt(x ** 2 + (y - 255) ** 2),
                          np.sqrt((x - 255) ** 2 + (y - 255) ** 2)])

        # 点光源分布权重矩阵
        for i in range(mask.shape[2]):
            for j in range(mask.shape[3]):
                mask[:, :, i, j] = np.sqrt((i - x) ** 2 + (j - y) ** 2) / max_len * (a - b) + b
        O = mask
    return np.float32(O)


# 根据给定点生成摩尔纹
def MoireGen(p_size, theta, center_x, center_y):
    z = np.zeros((p_size, p_size))
    for i in range(p_size):
        for j in range(p_size):
            z1 = 0.5 + 0.5 * math.cos(2 * math.pi * np.sqrt((i + 1 - center_x) ** 2 + (j + 1 - center_y) ** 2))
            z2 = 0.5 + 0.5 * math.cos(
                math.cos(theta / 180 * math.pi) * (j + 1) + math.sin(theta / 180 * math.pi) * (i + 1))
            z[i, j] = np.min([z1, z2])
    M = (z + 1) / 2
    return M


# 摩尔纹扭曲
def Moire_Distortion(embed_image):
    Z = np.zeros((embed_image.shape))
    for i in range(1):
        theta = np.random.randint(0, 180)

        # 在图像中随机采样一个点
        center_x = np.random.rand(1) * embed_image.shape[2]
        center_y = np.random.rand(1) * embed_image.shape[3]

        M = MoireGen(embed_image.shape[2], theta, center_x, center_y)
        Z[:, i, :, :] = M
    return np.float32(Z)


# 屏摄噪声
class ScreenShooting(nn.Module):

    def __init__(self):
        super(ScreenShooting, self).__init__()

    def forward(self, embed_image):
        device = embed_image.device

        # perspective transform
        noised_image = perspective(embed_image, device, 2)

        # Light Distortion，c 来随机选择点或线光源
        c = np.random.randint(0, 2)
        L = Light_Distortion(c, embed_image)
        Li = L.copy()

        # Moire Distortion
        Z = Moire_Distortion(embed_image) * 2 - 1
        Mo = Z.copy()

        # 0.85 * 光照扭曲 * 透视扭曲 + 0.15 * 摩尔纹扭曲
        noised_image = noised_image * torch.from_numpy(Li).to(device) * 0.85 + torch.from_numpy(Mo).to(device) * 0.15

        # 0.85 * 光照扭曲 * 透视扭曲 + 0.15 * 摩尔纹扭曲 + 剩余噪声(高斯模糊)
        noised_image = noised_image + 0.001 ** 0.5 * torch.randn(noised_image.size()).to(device)

        return noised_image


def salt_pepper_noise(images, prob):
    prob_zero = prob / 2
    prob_one = 1 - prob_zero
    rdn = torch.rand(images.shape).to(images.device)

    output = torch.where(rdn > prob_one, torch.zeros_like(images).to(images.device), images)
    output = torch.where(rdn < prob_zero, torch.ones_like(output).to(output.device), output)

    return output

class Salt_pepper(nn.Module):
    def __init__(self, prob):
        super(Salt_pepper, self).__init__()
        self.prob = prob

    def forward(self, images):
        return salt_pepper_noise(images, self.prob)



def gen_filters(size_x: int, size_y: int, dct_or_idct_fun: callable) -> np.ndarray:
    tile_size_x = 8
    filters = np.zeros((size_x * size_y, size_x, size_y))
    for k_y in range(size_y):
        for k_x in range(size_x):
            for n_y in range(size_y):
                for n_x in range(size_x):
                    filters[k_y * tile_size_x + k_x, n_y, n_x] = dct_or_idct_fun(n_y, k_y, size_y) * dct_or_idct_fun(
                        n_x,
                        k_x,
                        size_x)
    return filters


def get_jpeg_yuv_filter_mask(image_shape: tuple, window_size: int, keep_count: int):
    mask = np.zeros((window_size, window_size), dtype=np.uint8)

    index_order = sorted(((x, y) for x in range(window_size) for y in range(window_size)),
                         key=lambda p: (p[0] + p[1], -p[1] if (p[0] + p[1]) % 2 else p[1]))

    for i, j in index_order[0:keep_count]:
        mask[i, j] = 1

    return np.tile(mask, (int(np.ceil(image_shape[0] / window_size)),
                          int(np.ceil(image_shape[1] / window_size))))[0: image_shape[0], 0: image_shape[1]]


def dct_coeff(n, k, N):
    return np.cos(np.pi / N * (n + 1. / 2.) * k)


def idct_coeff(n, k, N):
    return (int(0 == n) * (- 1 / 2) + np.cos(
        np.pi / N * (k + 1. / 2.) * n)) * np.sqrt(1 / (2. * N))


def rgb2yuv(image_rgb, image_yuv_out):
    """ Transform the image from rgb to yuv """
    image_yuv_out[:, 0, :, :] = 0.299 * image_rgb[:, 0, :, :].clone() + 0.587 * image_rgb[:, 1, :, :].clone() + 0.114 * image_rgb[:, 2, :, :].clone() + 0.000035
    image_yuv_out[:, 1, :, :] = -0.14713 * image_rgb[:, 0, :, :].clone() + -0.28886 * image_rgb[:, 1, :, :].clone() + 0.436 * image_rgb[:, 2, :, :].clone() + 0.004179
    image_yuv_out[:, 2, :, :] = 0.615 * image_rgb[:, 0, :, :].clone() + -0.51499 * image_rgb[:, 1, :, :].clone() + -0.10001 * image_rgb[:, 2, :, :].clone() + 0.003960


def rgb2yuv_post(images):
    images[:, 0, :, :] = images[:, 0, :, :] - 0.000035
    images[:, 1, :, :] = images[:, 1, :, :] - 0.004179
    images[:, 2, :, :] = images[:, 2, :, :] - 0.003960
    return images


def yuv2rgb(image_yuv, image_rgb_out):
    """ Transform the image from yuv to rgb """
    image_rgb_out[:, 0, :, :] = image_yuv[:, 0, :, :].clone() + 1.13983 * image_yuv[:, 2, :, :].clone()
    image_rgb_out[:, 1, :, :] = image_yuv[:, 0, :, :].clone() + -0.39465 * image_yuv[:, 1, :, :].clone() + -0.58060 * image_yuv[:, 2, :, :].clone()
    image_rgb_out[:, 2, :, :] = image_yuv[:, 0, :, :].clone() + 2.03211 * image_yuv[:, 1, :, :].clone()


def yuv2rgb2(x):
    """ Transform the image from yuv to rgb """
    x[:, 1, :, :] = x[:, 1, :, :]
    x[:, 2, :, :] = x[:, 2, :, :]
    return x


class JpegCompression(nn.Module):
    def __init__(self, device, yuv_keep_weights=(25, 9, 9)):
        super(JpegCompression, self).__init__()
        self.device = device

        self.dct_conv_weights = torch.tensor(gen_filters(8, 8, dct_coeff), dtype=torch.float32).to(self.device)
        self.dct_conv_weights.unsqueeze_(1)
        self.idct_conv_weights = torch.tensor(gen_filters(8, 8, idct_coeff), dtype=torch.float32).to(self.device)
        self.idct_conv_weights.unsqueeze_(1)

        self.yuv_keep_weighs = yuv_keep_weights
        self.keep_coeff_masks = []

        self.jpeg_mask = None
        self.create_mask((1000, 1000))

    def create_mask(self, requested_shape):
        if self.jpeg_mask is None or requested_shape > self.jpeg_mask.shape[1:]:
            self.jpeg_mask = torch.empty((3,) + requested_shape, device=self.device)
            for channel, weights_to_keep in enumerate(self.yuv_keep_weighs):
                mask = torch.from_numpy(get_jpeg_yuv_filter_mask(requested_shape, 8, weights_to_keep))
                self.jpeg_mask[channel] = mask

    def get_mask(self, image_shape):
        if self.jpeg_mask.shape < image_shape:
            self.create_mask(image_shape)
        return self.jpeg_mask[:, :image_shape[1], :image_shape[2]].clone()

    def apply_conv(self, image, filter_type: str):
        if filter_type == 'dct':
            filters = self.dct_conv_weights
        elif filter_type == 'idct':
            filters = self.idct_conv_weights
        else:
            raise ValueError('Unknown filter_type value.')

        image_conv_channels = []

        for channel in range(image.shape[1]):
            # 克隆每个通道，避免原地修改
            image_yuv_ch = image[:, channel, :, :].clone().unsqueeze(1)
            image_conv = F.conv2d(image_yuv_ch, filters, stride=8)

            image_conv = image_conv.permute(0, 2, 3, 1)
            image_conv = image_conv.view(image_conv.shape[0], image_conv.shape[1], image_conv.shape[2], 8, 8)
            image_conv = image_conv.permute(0, 1, 3, 2, 4)
            image_conv = image_conv.contiguous().view(image_conv.shape[0],
                                                      image_conv.shape[1] * image_conv.shape[2],
                                                      image_conv.shape[3] * image_conv.shape[4])
            image_conv.unsqueeze_(1)

            image_conv_channels.append(image_conv)

        image_conv_stacked = torch.cat(image_conv_channels, dim=1)
        return image_conv_stacked

    def forward(self, images):
        # 将图像转换为 YUV 颜色空间
        image_yuv = rgb2yuv_post(images)

        # 对图像进行填充，使其能够被 8x8 块的 DCT 处理
        pad_height = (8 - image_yuv.shape[2] % 8) % 8
        pad_width = (8 - image_yuv.shape[3] % 8) % 8
        image_yuv = nn.ZeroPad2d((0, pad_width, 0, pad_height))(image_yuv)

        assert image_yuv.shape[2] % 8 == 0
        assert image_yuv.shape[3] % 8 == 0

        # 应用 DCT
        image_dct = self.apply_conv(image_yuv, 'dct')

        # 获取 JPEG 压缩掩码
        mask = self.get_mask(image_dct.shape[1:])

        # 对 DCT 结果进行掩码处理
        image_dct_mask = torch.mul(image_dct, mask)

        # 应用逆 DCT
        image_idct = self.apply_conv(image_dct_mask, 'idct')

        # 去掉填充的部分，返回处理后的图像
        noised_image = image_idct[:, :, :image_idct.shape[2] - pad_height, :image_idct.shape[3] - pad_width].clone()

        return noised_image


if __name__ == '__main__':
    # f = torch.randn([10, 1, 512, 512])
    # s = Scale_decode_eval()
    # ans = s(f)
    # print(ans.shape)

    # f = torch.randn([10, 1, 512, 512])
    # c = Crop()
    # ans = c(f)
    # print(ans.shape)

    # 高斯模糊测试
    img = torch.randn([10, 1, 512, 512]).cuda()
    ans = gaussian_blur(img)
    print(ans.shape)

    GN = GaussianNoise()
    ans2 = GN(img)
    print(ans2.shape)

    # sp = Salt_pepper(img)
    # print(sp.shape)

    import cv2

    image = cv2.imread("test_jc.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    image = torch.FloatTensor(image / 127.5 - 1.0).unsqueeze(0).permute(0, 3, 1, 2).to("cuda:0")

    jc = jpeg_compression_train(image)

    jc = jc.squeeze(0).permute(1, 2, 0).cpu().numpy()
    jc = np.clip((jc + 1.0) * 127.5, 0, 255).astype(np.uint8)
    cv2.imwrite("jced.jpg", cv2.cvtColor(jc, cv2.COLOR_YUV2BGR))

    # import cv2
    # import numpy
    # pic = cv2.imread('/home/zwl/cxm/VW_CNN_DTCWT/network/1.png')
    # pic = torch.tensor(pic).to(config.device)

    # img = torch.randn([2, 9, 512, 512, 1]).to(config.device)
    # ss = ScreenShooting()
    # noise = ss(img)
    # print(noise.shape)
    # print(noise.dtype)
    # noise = noise.cpu()
    # noise = noise.numpy()
    # cv2.imwrite("noise.jpg", noise)
