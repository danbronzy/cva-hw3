import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy import ndimage
from matplotlib import pyplot as plt
import cv2

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

def plotAndShow(img):
    plt.imshow(img, cmap="gray")
    plt.show()
    return

def LucasKanadeAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [3x3 numpy array] put your implementation here
    """

    # put your implementation here
    M = np.eye(3).astype(np.float32)
    p = recoverPfromM(M)
    template = It

    gradI_x = ndimage.sobel(It1, axis=1, mode='reflect')
    gradI_y = ndimage.sobel(It1, axis=0, mode='reflect')

    jacArray = np.zeros((It.shape[0], It.shape[1], 2, 6))
    for y in range(It.shape[0]):
        for x in range(It.shape[1]):
            jacArray[y, x, :, :] = getAffineJac(x,y)
    

    for _ in range(num_iters):
        #Warp new image and derivatives
        It1_warped_pre = warpImg(It1, M)
        gradI_x_warped = warpImg(gradI_x, M)
        gradI_y_warped = warpImg(gradI_y, M)

        #Making a mask so we only consider the intersectopm
        It1_mask_pre = warpImg(255*np.ones(It1.shape, dtype=np.uint8), M)
        _, It1_mask = cv2.threshold(It1_mask_pre, 254, 255, cv2.THRESH_BINARY)

        #mask everything
        template_masked = cv2.bitwise_and(template, template, mask=It1_mask)
        It1_warped = cv2.bitwise_and(It1_warped_pre, It1_warped_pre, mask=It1_mask)

        #calculate error image
        errImg = template_masked - It1_warped

        #Container for A vectors for each pixel
        As = np.zeros((It.shape[0], It.shape[1], 6))

        #calculate Hessian
        H = np.zeros((6, 6))
        for y in range(It.shape[0]):
            for x in range(It.shape[1]):
                #don't bother doing computations if outside It1_warped
                if (It1_mask[y,x] == 0):
                    continue
                
                #define jacobian
                thisJac = jacArray[y, x]

                gradI = np.reshape([gradI_x_warped[y, x], 
                                    gradI_y_warped[y, x]], 
                                    (1,2))

                A = np.dot(gradI, thisJac)
                As[y, x, :] = A
                H = H + np.dot(A.T, A)
                
        Hinv = np.linalg.inv(H)

        deltaP = np.zeros((p.shape))
        for y in range(It.shape[0]):
            for x in range(It.shape[1]):
                #don't bother doing computations if outside It1_warped
                if (It1_mask[y,x] == 0):
                    continue

                thisA = As[y, x, :].reshape((1,6))
                thisErr = errImg[y, x]
                deltaP = deltaP + np.dot(thisA.T, thisErr)
        
        deltaP = np.dot(Hinv, deltaP)

        p = p + deltaP
        M = recoverMfromP(p)
        dist = np.linalg.norm(deltaP)
        if (dist  < threshold):
            break
    
    return M
