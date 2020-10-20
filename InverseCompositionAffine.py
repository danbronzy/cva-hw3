import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy import ndimage
import cv2
from matplotlib import pyplot as plt

def recoverPfromM(M):
    M = M/M[2,2]
    p = np.array([M[0,0] - 1, M[0,1], M[0,2], M[1,0], M[1,1] - 1, M[1,2]]).reshape(6,1)
    return p

def recoverMfromP(p):
    M = np.array([[1 + p[0,0], p[1,0], p[2,0]], 
                [p[3,0], 1 + p[4,0], p[5,0]],
                [0,0,1]])
    return M

def warpImg(img, M):
    """
    :param img:  Image to be warped
    :param M: Affine transform matrix
    :return img_warped: warped image
    """
    M = M/M[2,2]
    warped = cv2.warpAffine(img, M[0:2,:], (img.shape[1], img.shape[0]), flags=cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR)

    return warped

def getAffineJac(x, y):
    return np.array([[x, y, 1, 0, 0, 0], [0, 0, 0, x, y, 1]])

def InverseCompositionAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [3x3 numpy array]
    """

    # put your implementation here
    template = It
    gradTemp_x = ndimage.sobel(template, axis=1, mode='reflect')
    gradTemp_y = ndimage.sobel(template, axis=0, mode='reflect')

    #compute jacobians and Hessian
    jacArray = np.array([ [getAffineJac(x, y) for x in range(template.shape[1])]for y in range(template.shape[0])])
    flatJac = jacArray.reshape((jacArray.shape[0] * jacArray.shape[1], jacArray.shape[2], jacArray.shape[3]))
    grads = np.vstack((gradTemp_x.flatten(), gradTemp_y.flatten()))
    sdi = np.array([np.dot(grads[:,ind], flatJac[ind]) for ind in range(flatJac.shape[0])])
    H = np.dot(sdi.T, sdi)
    Hinv = np.linalg.inv(H)

    M = np.eye(3)
    for itera in range(num_iters):
        print(itera)
        #Warp new image and derivatives
        It1_warped_pre = warpImg(It1, M)

        #Making a mask so we only consider the intersectopm
        It1_mask_pre = warpImg(255*np.ones(It1.shape, dtype=np.uint8), M)
        _, It1_mask = cv2.threshold(It1_mask_pre, 254, 255, cv2.THRESH_BINARY)

        #mask everything
        template_masked = cv2.bitwise_and(template, template, mask=It1_mask)
        It1_warped = cv2.bitwise_and(It1_warped_pre, It1_warped_pre, mask=It1_mask)

        #calculate error image
        errImg = (It1_warped - template_masked).flatten()
        deltaP = np.dot(Hinv, np.dot(sdi[np.flatnonzero(It1_mask)].T, 
                                      errImg[np.flatnonzero(It1_mask)])).reshape(6,1)
        
        deltaM = recoverMfromP(deltaP)
        M = np.dot(M, np.linalg.inv(deltaM))
        if np.linalg.norm(deltaP) <= threshold:
            break
    return M
