###########################################################################################
# @file     : TP1_Segmentation.py
# @authors  : Bilal Itani, Mehdi Kadi
###########################################################################################

import cv2
import argparse
import heapq
import copy

ap = argparse.ArgumentParser()

ap.add_argument("-p", "--path", default="data/highway/input",
                help="base path for the images directory")
ap.add_argument("-f", "--frames", default="1700",
                help="frame numbers")
ap.add_argument("-g", "--groundtruth", default="data/highway/groundtruth",
                help="Ground truth path")
ap.add_argument("-gtf", "--gtfrom", default="470",
                help="Ground truth from.")
ap.add_argument("-gtt", "--gtto", default="1700",
                help="Ground truth to.")

args = vars(ap.parse_args())


basePathInput = args["path"]
basePathGT = args["groundtruth"]
frameNumber = int(args["frames"])
gtfrom = int(args["gtfrom"])
gtto = int(args["gtto"])

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
    totalIoU = 0.0
    numberOfIntersections = 0
    maxIoU = -1
    minIoU = 999999999
    sortedIoU = []
    for i in range(1, frameNumber):
        frame = cv2.imread(basePathInput + "/in00" + getFileName(i) + ".jpg")
        cv2.imshow("Current frame", frame)
        fgmask = fgbg.apply(frame)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_DILATE, kernel)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        #fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

        #fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        cv2.imshow("After segmentation", fgmask)
        #Compare rectangles of interest GT v.s method using IoU
        if i >= gtfrom and i <= gtto:
            contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            gt = cv2.imread(basePathGT + "/gt00" + getFileName(i) + ".png", 0)
            gt_contours, gt_hierarchy = cv2.findContours(gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            clonedImg = copy.copy(fgmask)
            for cont in contours:
                x, y, w, h = cv2.boundingRect(cont)
                cv2.rectangle(clonedImg, (x, y), (x + w, y + h), (255, 255, 255), 5)
            cv2.imshow("Region of interest", clonedImg)
            for cont in gt_contours:
                x, y, w, h = cv2.boundingRect(cont)
                cv2.rectangle(gt, (x, y), (x + w, y + h), (255, 255, 255), 5)
            cv2.imshow("Ground Truth", gt)
            for gt_cont in gt_contours:
                gt_x, gt_y, gt_w, gt_h = cv2.boundingRect(gt_cont)
                bestMatch = []
                for cont in contours:
                    seg_x, seg_y, seg_w, seg_h = cv2.boundingRect(cont)
                    intersect = intersection((gt_x, gt_y, gt_w, gt_h), (seg_x,seg_y,seg_w,seg_h))
                    if len(intersect) > 0: #means theres an intersection
                        un = union((gt_x, gt_y, gt_w, gt_h), (seg_x,seg_y,seg_w,seg_h))
                        currentIoU = (float(intersect[2]) * float(intersect[3])) / (float(un[2]) * float(un[3]))
                        heapq.heappush(bestMatch, currentIoU)
                        heapq._heapify_max(bestMatch)
                        if minIoU > currentIoU:
                            minIoU = currentIoU
                        if maxIoU < currentIoU:
                            maxIoU = currentIoU
                if(len(bestMatch) > 0):
                    maxInHeap = heapq._heappop_max(bestMatch)
                    sortedIoU.append(maxInHeap)
                    totalIoU += maxInHeap
                    numberOfIntersections += 1
        cv2.waitKey(1)
    average = totalIoU / float(numberOfIntersections)
    sortedIoU.sort()
    median = -1
    if len(sortedIoU) % 2 == 0:
        median = (sortedIoU[int(len(sortedIoU) / 2)] + sortedIoU[int(len(sortedIoU) / 2) - 1])/2.0
    else:
        median = sortedIoU[int(len(sortedIoU)/2)]
    print("Median : " + str(median))
    print("Average : " + str(average))
    print("Total IoU : " + str(totalIoU))
    print("total intersections : " + str(numberOfIntersections))
    print("min IoU :" + str(minIoU))
    print("max IoU :" + str(maxIoU))
    cv2.destroyAllWindows()

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

processSegmentationBaselineCars()