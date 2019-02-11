import cv2
import argparse
import dlib
import time
import sys

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", default="data/face",
                help="base path for the dataset to analyze")
ap.add_argument("-f", "--frames", default="415",
                help="frame numbers")
ap.add_argument("-m", "--method", default="lbp",
                help="Method to use [lbp / hog]")
ap.add_argument("-gt", "--groundtruth", default="data/face/gt/groundtruth.txt",
                help="Method to use [lbp / hog]")

args = vars(ap.parse_args())

gtFile = None

def getFileName(n):
    if n < 10:
        return "0000000" + str(n)
    elif n < 100:
        return "000000" + str(n)
    elif n < 1000:
        return "00000" + str(n)

def readGroundTruth():
    global gtFile
    if gtFile is None:
        with open(args["groundtruth"]) as f:
            gtFile = f.readlines()
        # you may also want to remove whitespace characters like `\n` at the end of each line
        gtFile = [x.strip() for x in gtFile]

def getGroundTruthRectangle(frameNumber):
    global gtFile
    data = gtFile[frameNumber].split(',')
    return int(float(data[0])), int(float(data[1])), int(float(data[2])), int(float(data[3]))

def detect_faces_LBP(f_cascade, colored_img, scaleFactor=1.1):
    # just making a copy of image passed, so that passed image is not changed
    img_copy = colored_img.copy()

    # convert the test image to gray image as opencv face detector expects gray images
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)

    # let's detect multiscale (some images may be closer to camera than others) images
    faces = f_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=5);

    rectX = None
    rectY = None
    rectW = None
    rectH = None
    # go over list of faces and draw them as rectangles on original colored img
    for (x, y, w, h) in faces:
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
        rectX = x
        rectY = y
        rectW = w
        rectH = h
    return img_copy, rectX, rectY, rectW, rectH


#source : https://github.com/spmallick/learnopencv/blob/master/FaceDetectionComparison/face_detection_dlib_hog.py
def detectFaceDlibHog(detector, frame, inHeight=400, inWidth=0):
    frameDlibHog = frame.copy()
    frameHeight = frameDlibHog.shape[0]
    frameWidth = frameDlibHog.shape[1]
    if not inWidth:
        inWidth = int((frameWidth / frameHeight)*inHeight)

    scaleHeight = frameHeight / inHeight
    scaleWidth = frameWidth / inWidth

    frameDlibHogSmall = cv2.resize(frameDlibHog, (inWidth, inHeight))

    frameDlibHogSmall = cv2.cvtColor(frameDlibHogSmall, cv2.COLOR_BGR2RGB)
    faceRects = detector(frameDlibHogSmall, 0)
    #print(frameWidth, frameHeight, inWidth, inHeight)
    bboxes = []
    for faceRect in faceRects:
        cvRect = [int(faceRect.left()*scaleWidth), int(faceRect.top()*scaleHeight), int(faceRect.right()*scaleWidth) - int(faceRect.left()*scaleWidth),
                  int(faceRect.bottom()*scaleHeight) - int(faceRect.top()*scaleHeight)]
        bboxes.append(cvRect)
        cv2.rectangle(frameDlibHog, (cvRect[0], cvRect[1]), (cvRect[2] + cvRect[0], cvRect[3] + cvRect[1]), (0, 255, 0), int(round(frameHeight/150)), 4)
        print(str(cvRect[0]) + "\t" + str(cvRect[1]) + "\t" + str(cvRect[2]) + "\t" + str(cvRect[3]))
    return frameDlibHog, bboxes


def processHOG():
    hogFaceDetector = dlib.get_frontal_face_detector()
    readGroundTruth()
    totalIoU = 0.0
    maxIoU = -1
    minIoU = 999999999
    totalDetect = 0
    sortedIoU = []
    for i in range(1, int(args["frames"])):
        frame = cv2.imread("data/face" + "/" + getFileName(i) + ".jpg")
        outDlibHog, bboxes = detectFaceDlibHog(hogFaceDetector, frame)
        if len(bboxes) > 0 :
            x_gt, y_gt, gt_width, gt_height = getGroundTruthRectangle(i - 1)
            intersect = intersection((x_gt, y_gt, gt_width, gt_height), (bboxes[0][0], bboxes[0][1], bboxes[0][2], bboxes[0][3]))
            if len(intersect) > 0:
                un = union((x_gt, y_gt, gt_width, gt_height), (bboxes[0][0], bboxes[0][1], bboxes[0][2], bboxes[0][3]))
                currentIoU = (float(intersect[2]) * float(intersect[3])) / (float(un[2]) * float(un[3]))
                if minIoU > currentIoU:
                    minIoU = currentIoU
                if maxIoU < currentIoU:
                    maxIoU = currentIoU
                totalIoU += currentIoU
                sortedIoU.append(currentIoU)
        else:
            sortedIoU.append(0.0)
        totalDetect += 1
        cv2.imshow("DLIB HOG", outDlibHog)
        cv2.waitKey(10)
    average = totalIoU / float(totalDetect)
    median = -1
    sortedIoU.sort()
    if len(sortedIoU) % 2 == 0:
        median = (sortedIoU[int(len(sortedIoU) / 2)] + sortedIoU[int(len(sortedIoU) / 2) - 1]) / 2.0
    else:
        median = sortedIoU[int(len(sortedIoU) / 2)]
    print("Median : " + str(median))
    print("Average : " + str(average))
    print("Total IoU : " + str(totalIoU))
    print("total intersections : " + str(totalDetect))
    print("min IoU :" + str(minIoU))
    print("max IoU :" + str(maxIoU))
#From : https://github.com/informramiz/Face-Detection-OpenCV
def processLBP():
    lbp_face_cascade = cv2.CascadeClassifier('data/lbpcascade_frontalface.xml')
    readGroundTruth()
    totalIoU = 0.0
    maxIoU = -1
    minIoU = 999999999
    totalDetect = 0
    sortedIoU = []
    for i in range(1, int(args["frames"])):
        frame = cv2.imread("data/face" + "/" + getFileName(i) + ".jpg")
        faces_detected_img, rectX, rectY, rectW, rectH = detect_faces_LBP(lbp_face_cascade, frame)
        #print(str(rectX))
        if rectX is not None and rectY is not None and rectW is not None and rectH is not None:
            x_gt, y_gt, gt_width, gt_height = getGroundTruthRectangle(i - 1)
            intersect = intersection((x_gt, y_gt, gt_width, gt_height), (rectX, rectY, rectW, rectH))
            if len(intersect) > 0 : #theres an intersection
                un = union((x_gt, y_gt, gt_width, gt_height), (rectX, rectY, rectW, rectH))
                currentIoU = (float(intersect[2]) * float(intersect[3])) / (float(un[2]) * float(un[3]))
                if minIoU > currentIoU:
                    minIoU = currentIoU
                if maxIoU < currentIoU:
                    maxIoU = currentIoU
                totalIoU += currentIoU
                sortedIoU.append(currentIoU)
        else:
            sortedIoU.append(0.0)
        totalDetect += 1
        cv2.imshow("LBP_Experienced", faces_detected_img)
        cv2.waitKey(10)
    average = totalIoU / float(totalDetect)
    median = -1
    sortedIoU.sort()
    if len(sortedIoU) % 2 == 0:
        median = (sortedIoU[int(len(sortedIoU) / 2)] + sortedIoU[int(len(sortedIoU) / 2) - 1]) / 2.0
    else:
        median = sortedIoU[int(len(sortedIoU) / 2)]
    print("Median : " + str(median))
    print("Average : " + str(average))
    print("Total IoU : " + str(totalIoU))
    print("total intersections : " + str(totalDetect))
    print("min IoU :" + str(minIoU))
    print("max IoU :" + str(maxIoU))

def union(a, b):
    x = min(a[0], b[0])
    y = min(a[1], b[1])
    w = max(a[0] + a[2], b[0] + b[2]) - x
    h = max(a[1] + a[3], b[1] + b[3]) - y
    return (x, y, w, h)

def intersection(a, b):
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0] + a[2], b[0] + b[2]) - x
    h = min(a[1] + a[3], b[1] + b[3]) - y
    if w < 0 or h < 0:
        return ()
    return (x, y, w, h)

if args["method"] == "lbp":
    processLBP()
else:
    processHOG()