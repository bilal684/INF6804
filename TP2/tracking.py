import cv2
import argparse


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

def detect_faces(f_cascade, colored_img, scaleFactor=1.1):
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
        faces_detected_img, rectX, rectY, rectW, rectH = detect_faces(lbp_face_cascade, frame)
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
        totalDetect += 1
        cv2.imshow("LBP_Experienced", faces_detected_img)
        cv2.waitKey(10)
    average = totalIoU / float(totalDetect)
    median = -1
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

    #global gtFile
    #frameNumber = int(args["frames"])
    #readGroundTruth()

    #for i in range(2, frameNumber):
    #    x_init, y_init, width, height = getGroundTruthRectangle(i - 1)
    #    frame = cv2.imread(args["path"] + "/" + getFileName(i) + ".jpg")
        #gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #    cv2.rectangle(frame, (x_init, y_init), (x_init + width, y_init + height), (0, 255, 0), 1)
    #    cropped = frame[y_init:y_init+height, x_init:x_init+width]

    #    cv2.imshow("frame", frame)
    #    cv2.imshow("cropped", cropped)
        #cv2.imshow("frame", frame)
    #    cv2.waitKey(10)
    #frame = cv2.imread(basepath + "/00000001.jpg")

    #cv2.rectangle(frame, (130, 115), (130 + 85, 115 + 99), (0, 255, 0), 1)
    #cv2.imshow("frame 0", frame)
    #cv2.waitKey(0)


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