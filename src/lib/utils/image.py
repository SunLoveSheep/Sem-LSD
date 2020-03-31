# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Xingyi Zhou
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import cv2
import random


def get_gradient_magnitude(img_in, filter='scharr', threshold=0.3):
    def _get_pixel_safe_cv2(image, x, y):
        try:
            return image[y, x]

        except IndexError:
            return 0

    if filter == 'sobel':
        gx = [
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ]

        gy = [
            [1, 2, 1],
            [0, 0, 0],
            [-1, -2, -1]
        ]
    elif filter == 'scharr':
        gx = [
            [47, 0, -47],
            [162, 0, -162],
            [47, 0, -47]
        ]

        gy = [
            [47, 162, 47],
            [0, 0, 0],
            [-47, -162, -47]
        ]

    assert len(img_in.shape) == 2
    new_height, new_width = img_in.shape
    out_mag = np.zeros((new_height, new_width))
    # out_direct = np.zeros((new_height, new_width))

    for y in range(0, new_height):

        for x in range(0, new_width):
            gradient_y = (
                    gy[0][0] * _get_pixel_safe_cv2(img_in, x - 1, y - 1) +
                    gy[0][1] * _get_pixel_safe_cv2(img_in, x, y - 1) +
                    gy[0][2] * _get_pixel_safe_cv2(img_in, x + 1, y - 1) +
                    gy[2][0] * _get_pixel_safe_cv2(img_in, x - 1, y + 1) +
                    gy[2][1] * _get_pixel_safe_cv2(img_in, x, y + 1) +
                    gy[2][2] * _get_pixel_safe_cv2(img_in, x + 1, y + 1)
            )

            gradient_x = (
                    gx[0][0] * _get_pixel_safe_cv2(img_in, x - 1, y - 1) +
                    gx[0][2] * _get_pixel_safe_cv2(img_in, x + 1, y - 1) +
                    gx[1][0] * _get_pixel_safe_cv2(img_in, x - 1, y) +
                    gx[1][2] * _get_pixel_safe_cv2(img_in, x + 1, y) +
                    gx[2][0] * _get_pixel_safe_cv2(img_in, x - 1, y - 1) +
                    gx[2][2] * _get_pixel_safe_cv2(img_in, x + 1, y + 1)
            )
            gradient_magnitude = math.sqrt(pow(gradient_x, 2) + pow(gradient_y, 2))
            # gradient_direction = math.atan2(gradient_y, gradient_x)

            out_mag[y, x] = gradient_magnitude
            # out_direct[y, x] = gradient_direction

    out_mag = (out_mag - np.min(out_mag)) / (np.max(out_mag) - np.min(out_mag))
    out_mag[out_mag > threshold] = 1
    # out_direct = (out_direct - np.min(out_direct)) / (np.max(out_direct) - np.min(out_direct))
    # out_direct[out_mag == 0] = 0
    return out_mag


def flip(img):
    return img[:, :, ::-1].copy()


def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def crop(img, center, scale, output_size, rot=0):
    trans = get_affine_transform(center, scale, rot, output_size)

    dst_img = cv2.warpAffine(img,
                             trans,
                             (int(output_size[0]), int(output_size[1])),
                             flags=cv2.INTER_LINEAR)

    return dst_img


def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size
    
    a1  = 1
    b1  = (height + width)
    c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1  = (b1 + sq1) / 2
    
    a2  = 4
    b2  = 2 * (height + width)
    c2  = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2  = (b2 + sq2) / 2
    
    a3  = 4 * min_overlap
    b3  = -2 * min_overlap * (height + width)
    c3  = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3  = (b3 + sq3) / 2
    return min(r1, r2, r3)


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def gaussian2D_sep_rot(shape, sigma=1, r_xy=0):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    r_x, r_y = 1, 1
    default_r = 1.0
    # print(r_xy)
    if r_xy > 1:
        r_y = 8 * r_xy
        r_x = r_y / r_xy
        h = np.exp(-(r_x * x * x + default_r * r_y * y * y) / (2 * sigma * sigma))
    # if r_xy <= 1:
    else:
        r_x = 8 * 1 / r_xy
        r_y = r_x * r_xy
        h = np.exp(-(default_r * r_x * x * x + r_y * y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    # Need to make sure the largest possibility in each heatmap == 1.
    # Since the hm focal loss will be normalized by number of gt centers and
    # the gt centers is measured by if hm[pos] == 1
    # Normalize on generated Gaussian distribution. Since hm is per cate, the input hm may already has some values.
    domain = np.min(gaussian), np.max(gaussian)
    gaussian_normed = (gaussian - 0) / (domain[1] - 0)
    # gaussian_normed = gaussian

    # domain0 = np.min(gaussian_normed), np.max(gaussian_normed)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    # Seems causing problem in Precision results -10% precision.
    # left = int(left)
    # right = int(right)
    # top = int(top)
    # bottom = int(bottom)
    # radius = int(radius)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian_normed[radius - top:radius + bottom, radius - left:radius + right]
    # domain1 = np.min(gaussian_normed), np.max(gaussian_normed)
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    # domain2 = np.min(gaussian_normed), np.max(gaussian_normed)

    return heatmap


def draw_dense_reg(regmap, heatmap, center, value, radius, is_offset=False):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    value = np.array(value, dtype=np.float32).reshape(-1, 1, 1)
    dim = value.shape[0]
    reg = np.ones((dim, diameter * 2 + 1, diameter * 2 + 1), dtype=np.float32) * value
    if is_offset and dim == 2:
        delta = np.arange(diameter * 2 + 1) - radius
        reg[0] = reg[0] - delta.reshape(1, -1)
        reg[1] = reg[1] - delta.reshape(-1, 1)
    
    x, y = int(center[0]), int(center[1])
    
    height, width = heatmap.shape[0:2]
    
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    
    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_regmap = regmap[:, y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom,
                      radius - left:radius + right]
    masked_reg = reg[:, radius - top:radius + bottom,
                 radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        idx = (masked_gaussian >= masked_heatmap).reshape(
            1, masked_gaussian.shape[0], masked_gaussian.shape[1])
        masked_regmap = (1 - idx) * masked_regmap + idx * masked_reg
    regmap[:, y - top:y + bottom, x - left:x + right] = masked_regmap
    return regmap


def draw_msra_gaussian(heatmap, center, sigma):
    tmp_size = sigma * 3
    mu_x = int(center[0] + 0.5)
    mu_y = int(center[1] + 0.5)
    w, h = heatmap.shape[0], heatmap.shape[1]
    ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
    br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
    if ul[0] >= h or ul[1] >= w or br[0] < 0 or br[1] < 0:
      return heatmap
    size = 2 * tmp_size + 1
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    g_x = max(0, -ul[0]), min(br[0], h) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], w) - ul[1]
    img_x = max(0, ul[0]), min(br[0], h)
    img_y = max(0, ul[1]), min(br[1], w)
    heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
      heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]],
      g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
    return heatmap


def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def lighting_(data_rng, image, alphastd, eigval, eigvec):
    alpha = data_rng.normal(scale=alphastd, size=(3, ))
    image += np.dot(eigvec, eigval * alpha)


def blend_(alpha, image1, image2):
    image1 *= alpha
    image2 *= (1 - alpha)
    image1 += image2


def saturation_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    blend_(alpha, image, gs[:, :, None])


def brightness_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    image *= alpha


def contrast_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    blend_(alpha, image, gs_mean)


def color_aug(data_rng, image, eig_val, eig_vec, color_aug_var=0.4):
    functions = [brightness_, contrast_, saturation_]
    random.shuffle(functions)

    if len(image.shape) == 2:
        gs = image
    else:
        gs = grayscale(image)

    gs_mean = gs.mean()
    for f in functions:
        f(data_rng, image, gs, gs_mean, color_aug_var)
    lighting_(data_rng, image, 0.1, eig_val, eig_vec)
