###########################################################################################
# @file     : TP1_Segmentation.py
# @authors  : Bilal Itani, Mehdi Kadi
###########################################################################################

import numpy as np
import cv2
from skimage.measure import compare_ssim


basePathInput = "data/highway/input"
basePathGT = "data/highway/groundtruth"
frameNumber = 1700

#Source : https://www.pyimagesearch.com/2014/09/15/python-compare-two-images/
def getMSE(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err

#https://www.pyimagesearch.com/2017/06/19/image-difference-with-opencv-and-python/
def getSSIM(imageA, imageB):
    (score, diff) = compare_ssim(imageA, imageB, full=True)
    #diff = (diff * 255).astype("uint8")
    return score

def getFileName(n):
    if n < 10:
        return "000" + str(n)
    elif n < 100:
        return "00" + str(n)
    elif n < 1000:
        return "0" + str(n)
    else:
        return str(n)


def processSegmentationBaselineCars():
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    fgbg = cv2.createBackgroundSubtractorMOG2()
    min = 1
    max = -1
    for i in range(1, frameNumber):
        frame = cv2.imread(basePathInput + "/in00" + getFileName(i) + ".jpg")
        cv2.imshow("Current frame", frame)
        fgmask = fgbg.apply(frame)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        cv2.imshow('After Segmentation', fgmask)

        if i >= 470:
            gt = cv2.imread(basePathGT + "/gt00" + getFileName(i) + ".png", 0)
            cv2.imshow("Ground Truth", gt)
            score = getSSIM(fgmask, gt)
            if score < min :
                min = score
            if score > max :
                max = score
            print("SSIM: {}".format(score))
            #print("MSE: {}".format(getMSE(fgmask, gt)))
        #cv2.waitKey(33)
        cv2.waitKey(1)
    print("Min SSIM : " + str(min) + "\t" + "Max SSIM : " + str(max))
    cv2.destroyAllWindows()

processSegmentationBaselineCars()