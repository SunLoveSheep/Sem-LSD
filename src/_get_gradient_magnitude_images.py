import cv2
import math
import numpy as np
import os
from scipy import signal


img_src = '/Users/yisun/Desktop/Robotics_SemanticLines/data/KAIST_labeled/KAIST_seq38/'
# img_src = '/workspace/tangyang.sy/Robotics_SemanticLines/data/KAIST_seq39_0903/images/val_semantic_line/'
# img_src = '/workspace/tangyang.sy/pytorch_CV/test_imgs/KAIST_val_seq39_0816/Images/'
img_dst = img_src[:-1] + '_mag_nofiltered/'
if not os.path.exists(img_dst):
    os.makedirs(img_dst)
FILTER = 'scharr'
THRESHOLD = 0.075
if FILTER == 'sobel':
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
elif FILTER == 'scharr':
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


def get_gradient_magnitude(img_in, threshold=0.3):
    def _get_pixel_safe_cv2(image, x, y):
        try:
            return image[y, x]

        except IndexError:
            return 0

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


def get_gradient_magnitude_scipy(img_in):
    gradient_x = signal.convolve2d(img_in, gx, 'valid')
    gradient_y = signal.convolve2d(img_in, gy, 'valid')
    gradient_magnitude = np.sqrt(np.square(gradient_x) + np.square(gradient_y))
    gradient_magnitude = (gradient_magnitude - np.min(gradient_magnitude)) / \
                         (np.max(gradient_magnitude) - np.min(gradient_magnitude))
    # gradient_magnitude[gradient_magnitude > THRESHOLD] = 1.0
    # cv2.imshow('test', gradient_magnitude)
    # cv2.waitKey(0)
    return gradient_magnitude


img_lst = [f for f in sorted(os.listdir(img_src)) if '.png' in f]

for i, img_file in enumerate(img_lst):
    img = cv2.imread(img_src + img_file, 0)
    # img_mag = get_gradient_magnitude(img, threshold=THRESHOLD)
    img_mag = get_gradient_magnitude_scipy(img)
    # img_mag = filters.sobel(img)
    img_mag = img_mag * 255
    cv2.imwrite(img_dst + img_file[:-4] + '_mag.png', img_mag)
