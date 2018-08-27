# OpenCV (Open Source Computer Vision Library) used in this project is subject
# to license terms found in opencv_license.txt present in this distribution and at
# https://github.com/opencv/opencv_contrib/blob/master/LICENSE. If you do not
# agree to the license terms, do not download, install, copy or use the software

# importing stuff
import cv2 as cv
import argparse
import numpy as np
import sys
import os.path

# parsing arguments from terminal
ap = argparse.ArgumentParser(description = "YOLOv3")
ap.add_argument('--video', help = "Path to video file...")
ap.add_argument('--image', help = "Path to image file...")
args = ap.parse_args()

# threshold values for confidence and non-maxima suppression
confthresholdval = 0.5
nonmaxthresholdval = 0.4

# input height and width for the network (play around with 416x416, 608x608 etc.)
inputwidth = 320
inputheight = 320

# loading names of classes
classesdataset = "coco.names" #change this if not using a COCO pre-trained model
classes = None
with open(classesdataset, 'rt') as foo:
    classes = foo.read().rstrip('\n').split('\n')

# sampling 3 values for each of the Red, Green and Blue channels respectively
# from a uniform distribution ranging from 0-255 for all classes present in
# '<filename>.names'. So all the classes have a distinct color associated with them
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# details about model configuration and weights
modelconfiguration = 'yolov3.cfg'
modelweights = 'yolov3.weights'

# loading network configuration and weights
model = cv.dnn.readNetFromDarknet(modelconfiguration, modelweights)

# setting DNN backend to Open CV and using CPU
model.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
model.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

# function to get the names of output layers
def getoutputlayersnames(model):
    # getting names of all layers in the network
    layers_id = model.getLayerNames()
    # getting names of the layers with unconnected outputs
    return [layers_id[ind[0] - 1] for ind in model.getUnconnectedOutLayers()]

# function to draw the bounding box
def bounding_boxes(class_id, conf_value, left, top, right, bottom):
    # deciding color of the bounding box depending on the class
    color = COLORS[class_id]
    # drawing the bounding box
    cv.rectangle(frame, (left, top), (right, bottom), color, 2)

    # getting the label of the class along with its confidence value
    label = '%.2f' % conf_value
    if classes:
        assert(class_id < len(classes))
        label = '%s:%s' % (classes[class_id], label)

    # drawing a smaller filled white box on top of bounding box and displaying
    # the detected class name
    label_size, base_line = cv.getTextSize(label, cv.FONT_HERSHEY_DUPLEX, 0.3, 1)
    top = max(top, label_size[1])

    cv.rectangle(frame, (left, top - round(1.31*label_size[1])),
                 (left + round(1.31*label_size[0]), top + base_line),
                 (255, 255, 255), cv.FILLED)

    cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_DUPLEX, 0.4,
               (0,0,0), 1)

# function to apply non-maxima suppression
def postprocessing(frame, outs):
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    # scan through all the bounding boxes that are obtained from network output
    # and keep only the ones that are most likely to contain an object depending
    # on their confidence scores and then finally assign the class label with
    # the highest score to bounding box
    class_ids = []
    confidence_vals = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence_val = scores[class_id]
            if confidence_val > confthresholdval:
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                width = int(detection[2] * frame_width)
                height = int(detection[3] * frame_height)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                class_ids.append(class_id)
                confidence_vals.append(float(confidence_val))
                boxes.append([left, top, width, height])

    # applying non-maxima suppression to discard overlapping bounding boxes with
    # lower confidence values
    index_vals = cv.dnn.NMSBoxes(boxes, confidence_vals, confthresholdval,
                                 nonmaxthresholdval)
    for i in index_vals:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        bounding_boxes(class_ids[i], confidence_vals[i],
                       left, top, left + width, top + height)

# name of display window
window_name = 'YOLO (You Only Look Once)'
cv.namedWindow(window_name, cv.WINDOW_NORMAL)

# name of saved file in case of webcam input
output_file_yolo = "yolo_output_webcam.avi"

# for image input
if (args.image):
    # in case the image is not found at given location
    if not os.path.isfile(args.image):
        print("The given image file ", args.image, " is not present")
        sys.exit(1)
    # saving the image
    capture_frame = cv.VideoCapture(args.image)
    output_file_yolo = args.image[:-4]+'_yolo_output_image.jpg'

# for video input
elif (args.video):
    # in case the video is not found at given location
    if not os.path.isfile(args.video):
        print("The given video file ", args.video, " is not present")
        sys.exit(1)
    # processing frame of the video and saving it
    capture_frame = cv.VideoCapture(args.video)
    output_file_yolo = args.video[:-4]+'_yolo_output_video.avi'

# for webcam
else:
    capture_frame = cv.VideoCapture(0)

# if the input is not an image then use the OpenCV video writer to save the
# processed frames in sequence
if (not args.image):
    video_writer = cv.VideoWriter(output_file_yolo,
                                  cv.VideoWriter_fourcc('D','I','V','X'),
                                  30,
                                  (round(capture_frame.get(cv.CAP_PROP_FRAME_WIDTH)),
                                  round(capture_frame.get(cv.CAP_PROP_FRAME_HEIGHT))))

while cv.waitKey(1) < 0:
    # getting frames from the video
    frame_present, frame = capture_frame.read()

    # if frame is not present then stop execution
    if not frame_present:
        print("Processing operation complete")
        print("Output file is currently stored as -->", output_file_yolo)
        cv.waitKey(1500)
        break

    # converting input to blob that will be processed by network
    blob = cv.dnn.blobFromImage(frame, 1/255, (inputwidth, inputheight),
                                [0,0,0], 1, crop=False)

    # feeding the blob to network
    model.setInput(blob)

    # getting output from the network
    outs = model.forward(getoutputlayersnames(model))

    # using non-maxima suppression
    postprocessing(frame, outs)

    # displaying inference time on screen
    tick, _ = model.getPerfProfile()
    label = 'Inference time (in ms): %.2f' % (tick * 1000.0 / cv.getTickFrequency())
    cv.putText(frame, label, (0, 10), cv.FONT_HERSHEY_TRIPLEX, 0.35, (0, 0, 255))

    # writing finally processed frames with bounding boxes to output file
    if (args.image):
        # for image
        cv.imwrite(output_file_yolo, frame.astype(np.uint8));
    else:
        # for video
        video_writer.write(frame.astype(np.uint8))

    # displaying window
    cv.imshow(window_name, frame)
