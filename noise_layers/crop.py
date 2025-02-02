import torch.nn as nn
import numpy as np
import torch


def random_float(min, max):
    """
    Return a random number
    :param min:
    :param max:
    :return:
    """
    return np.random.rand() * (max - min) + min


def get_random_rectangle_inside(image, height_ratio_range, width_ratio_range):
    """
    Returns a random rectangle inside the image, where the size is random and is controlled by height_ratio_range and width_ratio_range.
    This is analogous to a random crop. For example, if height_ratio_range is (0.7, 0.9), then a random number in that range will be chosen
    (say it is 0.75 for illustration), and the image will be cropped such that the remaining height equals 0.75. In fact,
    a random 'starting' position rs will be chosen from (0, 0.25), and the crop will start at rs and end at rs + 0.75. This ensures
    that we crop from top/bottom with equal probability.
    The same logic applies to the width of the image, where width_ratio_range controls the width crop range.
    :param image: The image we want to crop
    :param height_ratio_range: The range of remaining height ratio
    :param width_ratio_range:  The range of remaining width ratio.
    :return: "Cropped" rectange with width and height drawn randomly height_ratio_range and width_ratio_range
    """
    image_height = image.shape[2]
    image_width = image.shape[3]

    remaining_height = int(np.rint(random_float(height_ratio_range[0], height_ratio_range[1]) * image_height))
    remaining_width = int(np.rint(random_float(width_ratio_range[0], width_ratio_range[0]) * image_width))

    if remaining_height == image_height:
        height_start = 0
    else:
        height_start = np.random.randint(0, image_height - remaining_height)

    if remaining_width == image_width:
        width_start = 0
    else:
        width_start = np.random.randint(0, image_width - remaining_width)

    return height_start, height_start+remaining_height, width_start, width_start+remaining_width


class Crop(nn.Module):
    """
    Randomly crops the image from top/bottom and left/right. The amount to crop is controlled by parameters
    heigth_ratio_range and width_ratio_range
    """
    def __init__(self, height_ratio_range, width_ratio_range):
        """

        :param height_ratio_range:
        :param width_ratio_range:
        """
        super(Crop, self).__init__()
        self.height_ratio_range = height_ratio_range
        self.width_ratio_range = width_ratio_range


    def forward(self, container, fake_image, secret_image, face_mask):
        # crop_rectangle is in form (from, to) where @from and @to are 2D points -- (height, width)

        h_start, h_end, w_start, w_end = get_random_rectangle_inside(container, self.height_ratio_range, self.width_ratio_range)
        padded_container = torch.zeros([4,3,256,256], device='cpu')
        padded_secret = torch.zeros([4,3,256,256], device='cpu')
        padded_fake = torch.zeros([4,3,256,256], device='cpu')
        padded_face_mask = torch.zeros([4,3,256,256], device='cpu')

        container_crop = container[
               :,
               :,
               h_start: h_end,
               w_start: w_end].clone()
        padded_container[:, :, h_start:h_end, w_start:w_end] = container_crop
               
        secret_crop = secret_image[
               :,
               :,
               h_start: h_end,
               w_start: w_end].clone()
        padded_secret[:, :, h_start:h_end, w_start:w_end] = secret_crop

        fake_crop = fake_image[
                      :,
                      :,
                      h_start: h_end,
                      w_start: w_end].clone()
        padded_fake[:, :, h_start:h_end, w_start:w_end] = fake_crop
               
        face_mask_crop = face_mask[
               :,
               :,
               h_start: h_end,
               w_start: w_end].clone()
        padded_face_mask[:, :, h_start:h_end, w_start:w_end] = face_mask_crop

        return padded_container, padded_fake, padded_secret, padded_face_mask