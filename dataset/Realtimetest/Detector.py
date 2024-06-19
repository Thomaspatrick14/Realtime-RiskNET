import torch
import torchvision
import getopt
import math
import numpy as np
import os
import PIL
import PIL.Image
import sys
import time
import matplotlib.pyplot as plt
import cv2
from sys import platform
from pathlib import Path
import json
from torchvision import transforms


target_classes = (0, 1, 2, 3, 5, 7)
combine_pairs = ((0, 1), (0, 3))
combined_classes = ("cyclist", "motorcyclist")
combined_class_ids = (212, 214)


def combine_bboxes(bboxes, class_ids):
    remaining_ids = list(range(len(bboxes)))
    new_bboxes = []
    new_class_ids = []

    for class_pair, new_class_id in zip(combine_pairs, combined_class_ids):
        ids1 = [i for i, cid in enumerate(class_ids) if cid == class_pair[0]]
        ids2 = [i for i, cid in enumerate(class_ids) if cid == class_pair[1]]
        if len(ids1) == 0 or len(ids2) == 0:
            continue

        iou_matrix = np.zeros((len(ids1), len(ids2)), dtype=float)
        for i1, id1 in enumerate(ids1):
            for i2, id2 in enumerate(ids2):
                l1, t1, r1, b1 = bboxes[id1] # l, t, r, b
                l2, t2, r2, b2 = bboxes[id2]
                a1 = (r1-l1) * (b1-t1)
                a2 = (r2-l2) * (b2-t2)
                intersection = max(0, (min(r2, r1) - max(l1, l2))) * max(0, (min(b1, b2) - max(t1, t2)))
                iou_matrix[i1, i2] = float(intersection) / (a1 + a2)
        ids1_for_ids2 = np.argmax(iou_matrix, axis=0)
        matched_pairs = [(ids1[ci1], ids2[ci2]) for ci2, ci1 in enumerate(ids1_for_ids2) if iou_matrix[ci1, ci2] > 0.01]
        # print(f"IOU matrix shape: {iou_matrix.shape}, iou matrix: {iou_matrix.flatten()}")
        # print(f"ids1_ids2 {ids1_for_ids2}, max values for each ids2: {[iou_matrix[ci1, ci2] for ci2, ci1 in enumerate(ids1_for_ids2)]}")
        # print(f"ids1 indices for ids2 indices: {ids1_for_ids2}, matched_pairs {matched_pairs}")

        for i1, i2 in matched_pairs:
            l1, t1, r1, b1 = bboxes[i1]
            l2, t2, r2, b2 = bboxes[i2]
            new_bbox = [min(l1, l2), min(t1, t2), max(r1, r2), max(b1, b2)]
            new_bboxes.append(new_bbox)
            new_class_ids.append(new_class_id)
            remaining_ids.remove(i1)
            remaining_ids.remove(i2)

    if len(new_bboxes) > 0:
        bboxes = [bbox for i, bbox in enumerate(bboxes) if i in remaining_ids]
        class_ids = [class_id for i, class_id in enumerate(class_ids) if i in remaining_ids]
        bboxes.extend(new_bboxes)
        class_ids.extend(new_class_ids)

    return bboxes, class_ids


def visualize(image, boxes, classes):
    image = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for i, box in enumerate(boxes):
        if classes[i] == 1:  # person
            color = (120, 170, 0)
            label = 'person'
        elif classes[i] == 2:  # bicycle
            color = (170, 120, 0)
            label = 'bicycle'
        elif classes[i] == 3:  # car
            color = (0, 0, 0)
            label = 'car'
        elif classes[i] == 4:  # motorcycle
            color = (255, 0, 0)
            label = 'motorcycle'
        elif classes[i] == 6:  # bus
            color = (0, 255, 0)
            label = 'bus'
        elif classes[i] == 8:  # truck
            color = (0, 0, 255)
            label = 'truck'
        elif classes[i] == 212:  # combined
            color = (170, 0, 120)
            label = 'cyclist'
        elif classes[i] == 214:  # combined
            color = (0, 120, 170)
            label = 'motorcyclist'
        else:
            raise ValueError("Non allowed classes were not filtered!")

        cv2.rectangle(
            image,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            color, 2
        )
        cv2.putText(image, label, (int(box[0]), int(box[1] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
                    lineType=cv2.LINE_AA)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image

    # cv2.imshow('Image', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def normalize_zero_one(data):
    min_value = np.min(data)
    max_value = np.max(data)
    normalized = ((data - min_value) / (max_value - min_value))
    return normalized


def keep_vehicles_only(boxes, classes):
    boxes_keep = []
    classes_keep = []
    keep_idx = [3, 4, 6, 8]
    for i in range(classes.shape[0]):
        if classes[i] in keep_idx:
            boxes_keep.append(boxes[i, :].tolist())
            classes_keep.append(classes[i])
    return np.array(boxes_keep), np.array(classes_keep)


def keep_roadusers_only(boxes, classes):
    boxes_keep = []
    classes_keep = []
    keep_idx = [0, 1, 2, 3, 5, 7]
    for i in range(classes.shape[0]):
        if classes[i] in keep_idx:
            boxes_keep.append(boxes[i, :].tolist())
            classes_keep.append(classes[i])
    return np.array(boxes_keep), np.array(classes_keep)


def predict(image, model, detection_threshold, cutoff_row=250):
    # cutoff_row = int(image.shape[0] * 0.955)
    # image[cutoff_row:, :, :] = 0 #if the bonnet is visible, it will be detected as a car
    # image = normalize_zero_one(image) #normalization doesnt seem to work, YOLO isn't detecting anything

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = image.transpose((2, 0, 1))
    # print(f"Shape of the image {image.shape}")
    # image = torch.from_numpy(image).to(device)
    # image = image.float().div(255.0).unsqueeze(0)

    t_start = time.time()
    with torch.no_grad():
        output = model(image)
    t_pred = time.time() - t_start
    print(f"Time taken: {t_pred}")
    
    ## get all the predicited class names

    # if image is loaded to cuda
    # pred_bboxes = output[0][:, :4].detach().cpu().numpy()
    # pred_scores = output[0][:, 4].detach().cpu().numpy()
    # pred_classes = output[0][:, 5].detach().cpu().numpy().astype(int)

    # if image is not loaded to cuda
    pred_bboxes = output.pred[0][:, :4].detach().cpu().numpy()
    pred_scores = output.pred[0][:, 4].detach().cpu().numpy()
    pred_classes = output.pred[0][:, 5].detach().cpu().numpy().astype(int)

    # get boxes and classes above the threshold score
    boxes = pred_bboxes[pred_scores >= detection_threshold].astype(np.int32) # size of (x,4) where x is the number of detections and y is Xmin, Ymin, Xmax, Ymax
    classes = pred_classes[pred_scores >= detection_threshold].astype(np.int32)

    # Scale the bounding boxes from 640x480 to 480x360
    # ratio = 360 / 480
    # boxes[:, :] = boxes[:, :] * ratio

    print(f"Shape of boxes: {boxes.shape}")
    return boxes, classes, t_pred

def detect(frames, model):
    THRESHOLD = 0.5

    detections = []

    # =============================
    # Resize the image to 640x480 (Yolo model expects dimensions which are a multiple of 32)
    # width, height = 640, 480    # for cuda loaded image
    width, height = 480, 360
    img = cv2.resize(frames, (width, height), interpolation=cv2.INTER_LINEAR)
    # =============================

    boxes, classes, t_pred = predict(img, model, detection_threshold=THRESHOLD)

    #boxes, classes = keep_vehicles_only(boxes, classes)
    boxes, classes = keep_roadusers_only(boxes, classes)

    classes = classes.tolist()
    boxes = boxes.tolist()
    boxes, classes = combine_bboxes(boxes, classes)

    frame_preds = {
        'Frame': 0,
        'Classes': classes,  # we won't use them (yet)
        'BBoxes': boxes
    }

    detections.append(frame_preds)
    try:
        print(f"\nTime to estimate detections for one frame: {t_pred:.3f} & {1/t_pred:.1f} FPS")
    except ZeroDivisionError:
        print(f"\nTime to estimate detections for one frame: {t_pred:.6f}")
    
          
    return detections
