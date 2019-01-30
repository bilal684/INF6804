# SOURCE : https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/


# import the necessary packages
import numpy as np
import argparse
import time
import cv2
import os
import heapq


def getFileName(n):
    if n < 10:
        return "000" + str(n)
    elif n < 100:
        return "00" + str(n)
    elif n < 1000:
        return "0" + str(n)
    else:
        return str(n)

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

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()

ap.add_argument("-p", "--path", default="data/highway/input",
                help="base path for the images directory")
ap.add_argument("-g", "--groundtruth", default="data/highway/groundtruth",
                help="Ground truth path")
ap.add_argument("-f", "--frames", default="1700",
                help="frame numbers")
ap.add_argument("-o", "--output", default="highway",
                help="Output video name.")
ap.add_argument("-y", "--yolo", default="yolo-coco",
                help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
                help="threshold when applyong non-maxima suppression")
ap.add_argument("-gtf", "--gtfrom", default="470",
                help="Ground truth from.")
ap.add_argument("-gtt", "--gtto", default="1700",
                help="Ground truth to.")
args = vars(ap.parse_args())

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                           dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# load our input image and grab its spatial dimensions
frameNumber = int(args["frames"])
gtfrom = int(args["gtfrom"])
gtto = int(args["gtto"])
writer = None
writer_gt = None
totalIoU = 0.0
numberOfIntersections = 0
maxIoU = -1
minIoU = 999999999
sortedIoU = []
for idx in range(1, frameNumber):
    print(idx)
    image = cv2.imread(args["path"] + "/in00" + getFileName(idx) + ".jpg")
    #clone_img = copy.copy(image)
    #cv2.imshow("Original image", image)
    (H, W) = image.shape[:2]

    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    # show timing information on YOLO
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))

    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > args["confidence"]:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
                            args["threshold"])
    gt = cv2.imread(args["groundtruth"] + "/gt00" + getFileName(idx) + ".png", 0)
    gt_contours, gt_hierarchy = cv2.findContours(gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cont in gt_contours:
        x, y, w, h = cv2.boundingRect(cont)
        cv2.rectangle(gt, (x, y), (x + w, y + h), (255, 255, 255), 5)
    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            #(x, y) = (boxes[i][0], boxes[i][1])
            x = boxes[i][0]
            y = boxes[i][1]
            w = boxes[i][2]
            h = boxes[i][3]
            #(w, h) = (boxes[i][2], boxes[i][3])

            # draw a bounding box rectangle and label on the image
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)
            if idx >= gtfrom and idx <= gtto:
                bestMatch = []
                for gt_cont in gt_contours:
                    gt_x, gt_y, gt_w, gt_h = cv2.boundingRect(gt_cont)
                    intersect = intersection((gt_x, gt_y, gt_w, gt_h), (x, y, w, h))
                    if len(intersect) > 0:  # means theres an intersection
                        un = union((gt_x, gt_y, gt_w, gt_h), (x, y, w, h))
                        currentIoU = (float(intersect[2]) * float(intersect[3])) / (float(un[2]) * float(un[3]))
                        heapq.heappush(bestMatch, currentIoU)
                        heapq._heapify_max(bestMatch)
                        #print("currentIoU : " + str(currentIoU))
                        #totalIoU += currentIoU
                        #numberOfIntersections += 1
                        if minIoU > currentIoU:
                            minIoU = currentIoU
                        if maxIoU < currentIoU:
                            maxIoU = currentIoU
                if len(bestMatch) > 0:
                    maxInHeap = heapq._heappop_max(bestMatch)
                    sortedIoU.append(maxInHeap)
                    totalIoU += maxInHeap
                    numberOfIntersections += 1

    if writer is None and writer_gt is None:
        # initialize our video writer
        #fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"] + ".mp4", 0x00000021, 30,
                                 (image.shape[1], image.shape[0]), True)
        writer_gt = cv2.VideoWriter(args["output"] + "_gt" + ".mp4", 0x00000021, 30,
                                 (gt.shape[1], gt.shape[0]), True)

        # some information on processing single frame
        if frameNumber > 0:
            elap = (end - start)
            print("[INFO] single frame took {:.4f} seconds".format(elap))
            print("[INFO] estimated total time to finish: {:.4f}".format(
                elap * frameNumber))

    cv2.waitKey(1)
    # write the output frame to disk
    writer.write(image)
    writer_gt.write(gt)
print("[INFO] cleaning up...")
average = totalIoU / float(numberOfIntersections)
sortedIoU.sort()
median = -1
if len(sortedIoU) % 2 == 0:
    median = (sortedIoU[int(len(sortedIoU) / 2)] + sortedIoU[int(len(sortedIoU) / 2) - 1]) / 2.0
else:
    median = sortedIoU[int(len(sortedIoU) / 2)]
print("Median : " + str(median))
print("Average : " + str(average))
print("Total IoU : " + str(totalIoU))
print("total intersections : " + str(numberOfIntersections))
print("min IoU :" + str(minIoU))
print("max IoU :" + str(maxIoU))
writer.release()
writer_gt.release()

    # show the output image
    #cv2.imshow("Classification image", clone_img)
    #cv2.waitKey(1)
