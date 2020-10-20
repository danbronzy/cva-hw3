import numpy as np
from InverseCompositionAffine import InverseCompositionAffine
from InverseCompositionAffine import warpImg
import cv2
from scipy import ndimage
import matplotlib.pyplot as plt
def SubtractDominantMotion(image1, image2, threshold, num_iters, tolerance):
    """
    :param image1: Images at time t
    :param image2: Images at time t+1
    :param threshold: used for LucasKanadeAffine
    :param num_iters: used for LucasKanadeAffine
    :param tolerance: binary threshold of intensity difference when computing the mask
    :return: mask: [nxm]
    """
    M = InverseCompositionAffine(image1, image2, threshold, int(num_iters))
    warped = warpImg(image2, M)
    diff = np.abs(image1 - warped)

    mask_pre = warpImg(255*np.ones(image2.shape, dtype=np.uint8), M)
    _, mask_img = cv2.threshold(mask_pre, 254, 255, cv2.THRESH_BINARY)

    _, thresh = cv2.threshold(diff, tolerance, 1.0, cv2.THRESH_BINARY)
    thresh_masked = cv2.bitwise_and(thresh, thresh, mask=mask_img)

    mask = ndimage.morphology.binary_dilation(thresh_masked)
    mask = ndimage.morphology.binary_dilation(mask)
    mask = ndimage.morphology.binary_dilation(mask)
    mask = ndimage.morphology.binary_closing(mask)

    return mask
