import sys
import cv2
import argparse
from random import randint

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", default="data/face",
                help="base path for the dataset to analyze")
ap.add_argument("-f", "--frames", default="415",
                help="frame numbers")
ap.add_argument("-gt", "--groundtruth", default="data/face/gt/groundtruth.txt",
                help="Method to use [lbp / hog]")
ap.add_argument("-v", "--video", help="video of the cups to analyze")

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
    if len(data) > 4:
        return int(float(data[2])), int(float(data[3])), int(float(data[6]) - float(data[2])), int(
            float(data[7]) - float(data[3]))
    else:
        return int(float(data[0])), int(float(data[1])), int(float(data[2])), int(float(data[3]))

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


def processCupVideoAnalysis():
    frameCounter = 0
    cap = cv2.VideoCapture(args["video"])
    success, frame = cap.read()
    resultFile = open("results.txt", "w")
    if not success:
        print('Failed to read video')
        sys.exit(1)
    frameCounter += 1
    bboxes = [(830,474,282,281), (1194,433,285,267)] #Two initial bounding boxes.
    colors = [(randint(64, 255), randint(64, 255), randint(64, 255)), (randint(64, 255), randint(64, 255), randint(64, 255))] #Two random colors

    print('Selected bounding boxes {}'.format(bboxes))

    multiTracker = cv2.MultiTracker_create()
    for bbox in bboxes:
        multiTracker.add(cv2.TrackerCSRT_create(), frame, bbox)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frameCounter += 1
        success, boxes = multiTracker.update(frame)

        # draw tracked objects
        for i, newbox in enumerate(boxes):
            p1 = (int(newbox[0]), int(newbox[1]))
            p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
            cv2.rectangle(frame, p1, p2, colors[i], 2, 1)
            resultFile.write(str(frameCounter) + " " + str(i + 1) + " " + str(int(newbox[0])) + " " + str(int(newbox[0] + newbox[2])) + " " + str(int(newbox[1])) + " " + str(int(newbox[1] + newbox[3])) + "\n")

        # show frame
        cv2.imshow('MultiTracker', frame)
        # quit on ESC button
        if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
            break
    resultFile.close()


def evaluateMethod():
    readGroundTruth()
    x_gt_init, y_gt_init, gt_init_width, gt_init_height = getGroundTruthRectangle(0)
    bbox = (x_gt_init, y_gt_init, gt_init_width, gt_init_height)
    color = (255,0,0)
    print('Selected bounding box {}'.format(bbox))
    totalIoU = 0.0
    maxIoU = -1
    minIoU = 999999999
    totalDetect = 2
    sortedIoUAllFrames = []
    sortedIoUWhenDetected = []
    frame = cv2.imread(args["path"] + "/" + getFileName(1) + ".jpg")
    multiTracker = cv2.MultiTracker_create()
    multiTracker.add(cv2.TrackerCSRT_create(), frame, bbox)

    for i in range(2, int(args["frames"])):
        frame = cv2.imread(args["path"] + "/" + getFileName(i) + ".jpg")
        success, boxes = multiTracker.update(frame)
        p1 = (int(boxes[0][0]), int(boxes[0][1]))
        p2 = (int(boxes[0][0] + boxes[0][2]), int(boxes[0][1] + boxes[0][3]))
        x_gt, y_gt, gt_width, gt_height = getGroundTruthRectangle(i - 1)

        intersect = intersection((x_gt, y_gt, gt_width, gt_height), (boxes[0][0], boxes[0][1], boxes[0][2], boxes[0][3]))
        if len(intersect) > 0:
            un = union((x_gt, y_gt, gt_width, gt_height), (boxes[0][0], boxes[0][1], boxes[0][2], boxes[0][3]))
            currentIoU = (float(intersect[2]) * float(intersect[3])) / (float(un[2]) * float(un[3]))
            if minIoU > currentIoU:
                minIoU = currentIoU
            if maxIoU < currentIoU:
                maxIoU = currentIoU
            totalIoU += currentIoU
            sortedIoUAllFrames.append(currentIoU)
            sortedIoUWhenDetected.append(currentIoU)
            totalDetect += 1
        else:
            sortedIoUAllFrames.append(0.0)
        cv2.rectangle(frame, p1, p2, color, 2, 1)
        cv2.rectangle(frame, (x_gt, y_gt), (x_gt + gt_width, y_gt + gt_height), (0,255,0), 4)
        cv2.imshow('CSRT', frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
            break
    averageAllFrames = totalIoU / float(int(args["frames"]))
    averageDetectedFrames = totalIoU / float(totalDetect)
    medianAllFrames = -1
    medianDetectedFrames = -1
    sortedIoUAllFrames.sort()
    sortedIoUWhenDetected.sort()
    if len(sortedIoUAllFrames) % 2 == 0:
        medianAllFrames = (sortedIoUAllFrames[int(len(sortedIoUAllFrames) / 2)] + sortedIoUAllFrames[int(len(sortedIoUAllFrames) / 2) - 1]) / 2.0
    else:
        medianAllFrames = sortedIoUAllFrames[int(len(sortedIoUAllFrames) / 2)]

    if len(sortedIoUWhenDetected) % 2 == 0:
        medianDetectedFrames = (sortedIoUWhenDetected[int(len(sortedIoUWhenDetected) / 2)] + sortedIoUWhenDetected[int(len(sortedIoUWhenDetected) / 2) - 1]) / 2.0
    else:
        medianDetectedFrames = sortedIoUWhenDetected[int(len(sortedIoUWhenDetected) / 2)]
    print("Median IoU all frames: " + str(medianAllFrames))
    print("Median IoU deteted frames: " + str(medianDetectedFrames))
    print("Average IoU all frames : " + str(averageAllFrames))
    print("Average IoU detected frames : " + str(averageDetectedFrames))
    print("Total IoU : " + str(totalIoU))
    print("total detections : " + str(totalDetect))
    print("total frames : " + args["frames"])
    print("min IoU :" + str(minIoU))
    print("max IoU :" + str(maxIoU))

if args["video"] is not None:
    processCupVideoAnalysis()
else:
    evaluateMethod()