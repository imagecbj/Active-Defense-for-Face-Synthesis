import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

import utils

class BasicBlock1(nn.Module):
	def __init__(self, in_channels, out_channels, r, drop_rate):
		super(BasicBlock1, self).__init__()

		self.downsample = None
		if (in_channels != out_channels):
			self.downsample = nn.Sequential(
				nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0,
						  stride=drop_rate, bias=False),
				nn.BatchNorm2d(out_channels)
			)

		self.left = nn.Sequential(
			nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1,
					  stride=drop_rate, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_channels),
		)

		self.se = nn.Sequential(
			nn.AdaptiveAvgPool2d((1, 1)),
			nn.Conv2d(in_channels=out_channels, out_channels=out_channels // r, kernel_size=1, bias=False),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_channels=out_channels, out_channels=out_channels // r, kernel_size=1, bias=False),
			nn.Sigmoid()
		)

	def forward(self, x):
		identity = x
		x = self.left(x)
		scale = self.se(x)
		x = x * scale

		if self.downsample is not None:
			identity = self.downsample(identity)

		x += identity
		x = F.relu(x)
		return x


class BottleneckBlock1(nn.Module):
	def __init__(self, in_channels, out_channels, r, drop_rate):
		super(BottleneckBlock1, self).__init__()

		self.downsample = None
		if (in_channels != out_channels):
			self.downsample = nn.Sequential(
				nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0,
						  stride=drop_rate, bias=False),
				nn.BatchNorm2d(out_channels)
			)

		self.left = nn.Sequential(
			nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
					  stride=drop_rate, padding=0, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, padding=0, bias=False),
			nn.BatchNorm2d(out_channels),
		)

		self.se = nn.Sequential(
			nn.AdaptiveAvgPool2d((1, 1)),
			nn.Conv2d(in_channels=out_channels, out_channels=out_channels // r, kernel_size=1, bias=False),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_channels=out_channels // r, out_channels=out_channels, kernel_size=1, bias=False),
			nn.Sigmoid()
		)

	def forward(self, x):
		identity = x
		x = self.left(x)
		scale = self.se(x)
		x = x * scale

		if self.downsample is not None:
			identity = self.downsample(identity)

		x += identity
		x = F.relu(x)
		return x


class SENet(nn.Module):
	'''
	SENet, with BasicBlock and BottleneckBlock
	'''

	def __init__(self, in_channels, out_channels, blocks, block_type="BottleneckBlock1", r=8, drop_rate=1):
		super(SENet, self).__init__()

		layers = [eval(block_type)(in_channels, out_channels, r, drop_rate)] if blocks != 0 else []
		for _ in range(blocks - 1):
			layer = eval(block_type)(out_channels, out_channels, r, drop_rate)
			layers.append(layer)

		self.layers = nn.Sequential(*layers)

	def forward(self, x):
		return self.layers(x)


class SENet_decoder(nn.Module):
	'''
	ResNet, with BasicBlock and BottleneckBlock
	'''

	def __init__(self, in_channels, out_channels, blocks, block_type="BottleneckBlock1", r=8, drop_rate=2):
		super(SENet_decoder, self).__init__()

		layers = [eval(block_type)(in_channels, out_channels, r, 1)] if blocks != 0 else []
		for _ in range(blocks - 1):
			layer1 = eval(block_type)(out_channels, out_channels, r, 1)
			layers.append(layer1)
			layer2 = eval(block_type)(out_channels, out_channels * drop_rate, r, drop_rate)
			out_channels *= drop_rate
			layers.append(layer2)

		self.layers = nn.Sequential(*layers)

	def forward(self, x):
		return self.layers(x)
class ConvTBNRelu(nn.Module):
	"""
	A sequence of TConvolution, Batch Normalization, and ReLU activation
	"""

	def __init__(self, channels_in, channels_out, stride=2):
		super(ConvTBNRelu, self).__init__()

		self.layers = nn.Sequential(
			nn.ConvTranspose2d(channels_in, channels_out, kernel_size=2, stride=stride, padding=0),
			nn.BatchNorm2d(channels_out),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		return self.layers(x)


class ExpandNet(nn.Module):
	'''
	Network that composed by layers of ConvTBNRelu
	'''

	def __init__(self, in_channels, out_channels, blocks):
		super(ExpandNet, self).__init__()

		layers = [ConvTBNRelu(in_channels, out_channels)] if blocks != 0 else []
		for _ in range(blocks - 1):
			layer = ConvTBNRelu(out_channels, out_channels)
			layers.append(layer)

		self.layers = nn.Sequential(*layers)

	def forward(self, x):
		return self.layers(x)
class ConvBNRelu(nn.Module):
	"""
	A sequence of Convolution, Batch Normalization, and ReLU activation
	"""

	def __init__(self, channels_in, channels_out, stride=1):
		super(ConvBNRelu, self).__init__()

		self.layers = nn.Sequential(
			nn.Conv2d(channels_in, channels_out, 3, stride, padding=1),
			nn.BatchNorm2d(channels_out),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		return self.layers(x)


class ConvNet(nn.Module):
	'''
	Network that composed by layers of ConvBNRelu
	'''

	def __init__(self, in_channels, out_channels, blocks):
		super(ConvNet, self).__init__()

		layers = [ConvBNRelu(in_channels, out_channels)] if blocks != 0 else []
		for _ in range(blocks - 1):
			layer = ConvBNRelu(out_channels, out_channels)
			layers.append(layer)

		self.layers = nn.Sequential(*layers)

	def forward(self, x):
		return self.layers(x)
class Conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation='relu', is_bn=False, strides=1):
        super(Conv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.activation = activation
        self.strides = strides

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, strides, int((kernel_size - 1) / 2))
        self.is_bn = is_bn
        if self.is_bn:
            self.bn = nn.BatchNorm2d(out_channels)
        # default: using he_normal as the kernel initializer
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, inputs):
        outputs = self.conv(inputs)
        if self.is_bn:
            outputs = self.bn(outputs)
        if self.activation is not None:
            if self.activation == 'relu':
                outputs = nn.ReLU(inplace=True)(outputs)
            else:
                raise NotImplementedError
        return outputs


class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, activation='relu', is_bn=False, strides=1):
        super(BasicBlock, self).__init__()
        self.is_bn = is_bn

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = Conv2D(in_channels, out_channels, kernel_size=kernel_size, activation=None, is_bn=False,
                            strides=strides)
        if self.is_bn:
            self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        # self.conv2 = Conv2D(in_channels, out_channels, kernel_size=kernel_size, activation=None, is_bn=False, strides=1)
        # if self.is_bn:
        # self.bn2 = nn.BatchNorm2d(out_channels)
        if self.is_bn:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=strides, bias=False),
                nn.BatchNorm2d(out_channels))
        else:
            self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=strides, bias=False)
        self.strides = strides

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        if self.is_bn:
            out = self.bn1(out)
        # out = self.relu(out)

        # out = self.conv2(out)
        # if self.is_bn:
        # out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out



class DoubleConv(nn.Module):
    """(conv => BN => ReLU) * 3"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class DoubleConv1(nn.Module):
    """(conv => BN => ReLU) * 3"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv1, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = self.conv(x1)
        x = torch.cat([x, x2], dim=1)
        return x
class Down1(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super(Down1, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up1(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up1, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv1(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = self.conv(x1)
        x = torch.cat([x, x2], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class SCSEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SCSEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.channel_excitation = nn.Sequential(nn.Linear(channel, int(channel // reduction)),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(int(channel // reduction), channel),
                                                nn.Sigmoid())

        self.spatial_se = nn.Sequential(nn.Conv2d(channel, 1, kernel_size=1,
                                                  stride=1, padding=0, bias=False),
                                        nn.Sigmoid())

    def forward(self, x):
        bahs, chs, _, _ = x.size()

        # Returns a new tensor with the same data as the self tensor but of a different size.
        chn_se = self.avg_pool(x).view(bahs, chs)
        chn_se = self.channel_excitation(chn_se).view(bahs, chs, 1, 1)
        chn_se = torch.mul(x, chn_se)

        spa_se = self.spatial_se(x)
        spa_se = torch.mul(x, spa_se)
        return torch.add(chn_se, 1, spa_se)

class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, reduction, stride, attention=None):
        super(BottleneckBlock, self).__init__()

        self.change = None
        if (in_channels != out_channels or stride != 1):
            self.change = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0,
                          stride=stride, bias=False),
                nn.InstanceNorm2d(out_channels)
            )

        self.left = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                      stride=stride, padding=0, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, padding=0, bias=False),
            nn.InstanceNorm2d(out_channels)
        )

        if attention == 'se':
            print('SEAttention')
            self.attention = SCSEBlock(channel=out_channels, reduction=reduction)

        else:
            print('None Attention')
            self.attention = nn.Identity()

    def forward(self, x):
        identity = x
        x = self.left(x)
        x = self.attention(x)

        if self.change is not None:
            identity = self.change(identity)

        x += identity
        x = F.relu(x)
        return x


class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, blocks=1, block_type="BottleneckBlock", reduction=8, stride=1, attention=None):
        super(ResBlock, self).__init__()

        layers = [eval(block_type)(in_channels, out_channels, reduction, stride, attention=attention)] if blocks != 0 else []
        for _ in range(blocks - 1):
            layer = eval(block_type)(out_channels, out_channels, reduction, 1, attention=attention)
            layers.append(layer)

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)




class ENet2(nn.Module):
	'''
	Insert a watermark into an image
	'''

	def __init__(self, H = 256, W = 256, message_length = 256, blocks=4, channels=64):
		super(ENet2, self).__init__()
		self.H = H
		self.W = W

		message_convT_blocks = int(np.log2(H // int(np.sqrt(message_length))))
		message_se_blocks = max(blocks - message_convT_blocks, 1)

		self.image_pre_layer = ConvBNRelu(3, channels)
		self.image_first_layer = SENet(channels, channels, blocks=blocks)

		self.message_pre_layer = nn.Sequential(
			ConvBNRelu(1, channels),
			ExpandNet(channels, channels, blocks=message_convT_blocks),
			SENet(channels, channels, blocks=message_se_blocks),
		)

		self.message_first_layer = SENet(channels, channels, blocks=blocks)

		self.after_concat_layer = ConvBNRelu(2 * channels, channels)

		self.final_layer = nn.Conv2d(channels + 3, 3, kernel_size=1)

	def forward(self, image, message):
		# first Conv part of Encoder
		image_pre = self.image_pre_layer(image)
		intermediate1 = self.image_first_layer(image_pre)

		# Message Processor
		size = int(np.sqrt(message.shape[1]))
		message_image = message.view(-1, 1, size, size)
		message_pre = self.message_pre_layer(message_image)
		intermediate2 = self.message_first_layer(message_pre)

		# concatenate
		concat1 = torch.cat([intermediate1, intermediate2], dim=1)

		# second Conv part of Encoder
		intermediate3 = self.after_concat_layer(concat1)

		# skip connection
		concat2 = torch.cat([intermediate3, image], dim=1)

		# last Conv part of Encoder
		output = self.final_layer(concat2)

		return output




class RNet2(nn.Module):
	'''
	Decode the encoded image and get message
	'''

	def __init__(self, H = 256, W = 256, message_length = 256, blocks=4, channels=64):
		super(RNet2, self).__init__()

		stride_blocks = int(np.log2(H // int(np.sqrt(message_length))))
		keep_blocks = max(blocks - stride_blocks, 0)

		self.first_layers = nn.Sequential(
			ConvBNRelu(3, channels),
			SENet_decoder(channels, channels, blocks=stride_blocks + 1),
			ConvBNRelu(channels * (2 ** stride_blocks), channels),
		)
		self.keep_layers = SENet(channels, channels, blocks=keep_blocks)

		self.final_layer = ConvBNRelu(channels, 1)

	def forward(self, noised_image):
		x = self.first_layers(noised_image)
		x = self.keep_layers(x)
		x = self.final_layer(x)
		x = x.view(x.shape[0], -1)
		return x



def dsl_net(encoded_image, fake_image, secret_image, face_mask, args, global_step, Crop_layer, Resize_layer):
    encoded_image = encoded_image.cpu()
    # encoded_image = (encoded_image + 1.0)/2.0
    # secret_image = (secret_image + 1.0)/2.0
    sh = encoded_image.size()
    ramp_fn = lambda ramp: np.min([global_step / ramp, 1.])

    rnd_bri = ramp_fn(args.rnd_bri_ramp) * args.rnd_bri
    rnd_hue = ramp_fn(args.rnd_hue_ramp) * args.rnd_hue
    rnd_brightness = utils.get_rnd_brightness_torch(rnd_bri, rnd_hue, args.batch_size)  # [batch_size, 3, 1, 1]
    jpeg_quality = 100. - torch.rand(1)[0] * ramp_fn(args.jpeg_quality_ramp) * (100. - args.jpeg_quality)
    rnd_noise = torch.rand(1)[0] * ramp_fn(args.rnd_noise_ramp) * args.rnd_noise

    contrast_low = 1. - (1. - args.contrast_low) * ramp_fn(args.contrast_ramp)
    contrast_high = 1. + (args.contrast_high - 1.) * ramp_fn(args.contrast_ramp)
    contrast_params = [contrast_low, contrast_high]

    rnd_sat = torch.rand(1)[0] * ramp_fn(args.rnd_sat_ramp) * args.rnd_sat

    # Resize
    resize_ratio_min = 1. - args.rnd_resize * ramp_fn(args.rnd_resize_ramp)
    resize_ratio_max = 1. + args.rnd_resize * ramp_fn(args.rnd_resize_ramp)
    Resize_layer.resize_ratio_min = resize_ratio_min
    Resize_layer.resize_ratio_max = resize_ratio_max
    encoded_image, fake_image, secret_image, face_mask = Resize_layer(encoded_image, fake_image, secret_image, face_mask)

    # Resize back to 252x256
    encoded_image = F.interpolate(encoded_image, size=(args.resolution, args.resolution), mode='bilinear')
    fake_image = F.interpolate(fake_image, size=(args.resolution, args.resolution), mode='bilinear')
    secret_image = F.interpolate(secret_image, size=(args.resolution, args.resolution), mode='bilinear')
    face_mask = F.interpolate(face_mask, size=(args.resolution, args.resolution), mode='bilinear')
    face_mask[face_mask > 0.5] = 1.0

    # Crop
    ratio_range = 1. - args.rnd_crop * ramp_fn(args.rnd_crop_ramp)
    ratio_range = 1. - args.rnd_crop * ramp_fn(args.rnd_crop_ramp)
    Crop_layer.height_ratio_range = [ratio_range, 1.0]
    Crop_layer.width_ratio_range = [ratio_range, 1.0]
    encoded_image, fake_image, secret_image, face_mask = Crop_layer(encoded_image, fake_image, secret_image, face_mask)

    # blur the code borrowed from https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/10
    kernel_size = 5
    dim = 2
    kernel_size = [kernel_size] * dim
    sigma = np.random.randint(2, 5, size=1)[0]
    sigma = [sigma] * dim

    # The gaussian kernel is the product of the
    # gaussian function of each dimension.
    kernel = 1
    meshgrids = torch.meshgrid(
        [
            torch.arange(size, dtype=torch.float32)
            for size in kernel_size
        ]
    )
    for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
        mean = (size - 1) / 2
        kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                  torch.exp(-((mgrid - mean) / std) ** 2 / 2)

    # Make sure sum of values in gaussian kernel equals 1.
    kernel = kernel / torch.sum(kernel)

    # Reshape to depthwise convolutional weight
    kernel = kernel.view(1, 1, *kernel.size())
    kernel = kernel.repeat(3, *[1] * (kernel.dim() - 1))

    # f = utils.random_blur_kernel(probs=[.25, .25], N_blur=kernel_size, sigrange_gauss=[1., 3.], sigrange_line=[.25, 1.], wmin_line=3)
    if args.is_cuda:
        kernel = kernel.cuda()

    if np.random.rand() < args.blur_prob:
        encoded_image = F.conv2d(encoded_image, kernel, bias=None, padding=int((kernel_size[0] - 1) / 2), groups=3)

    # noise
    noise = torch.normal(mean=0, std=rnd_noise, size=encoded_image.size(), dtype=torch.float32)
    if args.is_cuda:
        noise = noise.cuda()
    encoded_image = encoded_image + noise
    encoded_image = torch.clamp(encoded_image, 0, 1)

    # contrast & brightness
    # contrast_scale = torch.Tensor(encoded_image.size()[0]).uniform_(contrast_params[0], contrast_params[1])
    # contrast_scale = contrast_scale.reshape(encoded_image.size()[0], 1, 1, 1)
    # if args.is_cuda:
    #     contrast_scale = contrast_scale.cuda()
    #     rnd_brightness = rnd_brightness.cuda()
    # encoded_image = encoded_image * contrast_scale
    # encoded_image = encoded_image + rnd_brightness
    # encoded_image = torch.clamp(encoded_image, 0, 1)

    # saturation
    # sat_weight = torch.FloatTensor([.3, .6, .1]).reshape(1, 3, 1, 1)
    # if args.is_cuda:
    #     sat_weight = sat_weight.cuda()
    # encoded_image_lum = torch.mean(encoded_image * sat_weight, dim=1).unsqueeze_(1)
    # encoded_image = (1 - rnd_sat) * encoded_image + rnd_sat * encoded_image_lum

    # jpeg
    encoded_image = encoded_image.reshape([-1, 3, args.resolution, args.resolution])
    if not args.no_jpeg:
        encoded_image = utils.jpeg_compress_decompress(encoded_image, args.is_cuda, rounding=utils.round_only_at_0,
                                                       quality=jpeg_quality)

    # encoded_image = (encoded_image - 0.5) * 2.0
    # secret_image = (secret_image - 0.5) * 2.0
    encoded_image = encoded_image.cuda()
    fake_image = fake_image.cuda()
    secret_image = secret_image.cuda()
    face_mask = face_mask.cuda()
    return encoded_image, fake_image, secret_image, face_mask
