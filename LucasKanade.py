import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy import ndimage
from matplotlib import pyplot as plt

def warpImg(img, rect, p):
    """
    :param img:  Image to be warped
    :param rect: original rectangular area 
    :param p: movement vector
    :return img_warped: warped image (rectangle of image offset by movement vector) 
    """
    
    origXdim = int(rect[2] - rect[0])
    origYdim = int(rect[3] - rect[1])
    interpolator = RectBivariateSpline(range(img.shape[0]), range(img.shape[1]), img)
    newX1 = rect[0] + p[0]
    newY1 = rect[1] + p[1]
    newX2 = rect[2] + p[0]
    newY2 = rect[3] + p[1]

    # xRange = np.arange(newX1, newX2 + 1, 1)
    # yRange = np.arange(newY1, newY2 + 1, 1)
    xRange = np.linspace(newX1, newX2, origXdim + 1)
    yRange = np.linspace(newY1, newY2, origYdim + 1)

    img_warped = interpolator(yRange, xRange)

    return img_warped

def LucasKanade(It, It1, rect, threshold, num_iters, p0=np.zeros(2)):
    """
    :param It: template image
    :param It1: Current image
    :param rect: Current position of the car (top left, bot right coordinates)
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :param p0: Initial movement vector [dp_x0, dp_y0]
    :return: p: movement vector [dp_x, dp_y]
    """

    # Get actual template
    template = warpImg(It, rect, np.array(np.zeros(2)))
    # Get full image gradient
    gradI_x = ndimage.sobel(It1, axis=1, mode='reflect')
    gradI_y = ndimage.sobel(It1, axis=0, mode='reflect')

    p = p0.reshape((2,1))
    
    for _ in range(num_iters):
        # Warp It1
        It1_warped = warpImg(It1, rect, p)
        
        errImg = template - It1_warped
        
        # Warp to get gradient of warped image
        gradI_x_warped = warpImg(gradI_x, rect, p)
        gradI_y_warped = warpImg(gradI_y, rect, p)

        #jacobian is Identity matrix
        jac = np.array([[1,0],[0,1]])

        H = np.zeros((2,2))
        for z in zip(gradI_x_warped.flatten(), gradI_y_warped.flatten()):
            gradI = np.reshape([z[0], z[1]], (1,2))
            mult = np.dot(gradI, jac)
            H = H + np.dot(mult.T, mult)

        Hinv = np.linalg.inv(H)

        deltaP = np.zeros((2,1))

        for z in zip(gradI_x_warped.flatten(), gradI_y_warped.flatten(), errImg.flatten()):
            gradI = np.reshape([z[0], z[1]], (1,2))
            A = np.dot(gradI, jac)
            b = z[2]
            deltaP = deltaP + np.dot(A.T, b)
        
        deltaP = np.dot(Hinv, deltaP)
        
        dist = np.linalg.norm(deltaP)

        p = p + deltaP
        if (dist < threshold):
            break
    return p.T
