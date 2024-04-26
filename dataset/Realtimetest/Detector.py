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


target_classes = (1, 2, 3, 4, 6, 8)
combine_pairs = ((1, 2), (1, 4))
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
    keep_idx = [1, 2, 3, 4, 6, 8]
    for i in range(classes.shape[0]):
        if classes[i] in keep_idx:
            boxes_keep.append(boxes[i, :].tolist())
            classes_keep.append(classes[i])
    return np.array(boxes_keep), np.array(classes_keep)


def predict(image, model, detection_threshold, cutoff_row=250):
    cutoff_row = int(image.shape[0] * 0.955)
    image[cutoff_row:, :, :] = 0
    image = image.transpose((2, 0, 1))  # Faster R-CNN requires C, H, W
    image = normalize_zero_one(image)
    image = torch.Tensor(image)
    if torch.cuda.is_available():
        image = image.cuda()
    t_start = time.time()
    output = model([image])
    t_pred = time.time() - t_start

    # get all the predicited class names
    pred_classes = output[0]['labels'].cpu().numpy()  # [coco_names[i] for i in output[0]['labels'].cpu().numpy()]
    pred_scores = output[0]['scores'].detach().cpu().numpy()
    pred_bboxes = output[0]['boxes'].detach().cpu().numpy()

    # get boxes and classes above the threshold score
    boxes = pred_bboxes[pred_scores >= detection_threshold].astype(np.int32)
    classes = pred_classes[pred_scores >= detection_threshold].astype(np.int32)

    return boxes, classes, t_pred


coco_names = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def detect(frames, img_folder, model):
    # if platform == "win32":
    #     img_folder = Path(r"C:/JP/TUe/2nd year Internship and thesis/Internship/Idiada/car_to_bicycle_turning/2023-01-23-16-55-30_filtered_cropped")
    # else:
    #     img_folder = Path("/home/developer/Projects/TUe/AITHENA/Data/risknet_clips")   

    # FPS = 10 # 30
    THRESHOLD = 0.9
    show_predictions = False  # put this on true for the first time you run a clip, in order to determine the cut value

    if show_predictions:
        out_dir = Path(img_folder, "detections")
        os.makedirs(out_dir, exist_ok=True)

    # img_names = sorted(os.listdir(img_folder))

    # print(f"Loading the model")
    # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # if torch.cuda.is_available():
    #     print("CUDA available: loading the model on the GPU")
    #     model = model.cuda()
    # print(f"Model loaded")
    # model.eval()

    detections = []
    pred_times = []
    t = time.time()
    
    if isinstance(frames, list):
        for i, frame in enumerate(frames):
            # name = img_names[i]
            # path_img = img_folder / name
            img = np.array(frame)
            # OPTION 1. resize image
            target_w = 480
            target_h = 360
            img = cv2.resize(img, dsize=(target_w, target_h))
            # =============================
            
            boxes, classes, t_pred = predict(img, model, detection_threshold=THRESHOLD)
            pred_times.append(t_pred)

            # if i%25 == 0:
            #     print(f"{i} took {t_pred:.1f} s to create, including loading the images")
            #boxes, classes = keep_vehicles_only(boxes, classes)
            boxes, classes = keep_roadusers_only(boxes, classes)

            classes = classes.tolist()
            boxes = boxes.tolist()
            boxes, classes = combine_bboxes(boxes, classes)

            if show_predictions:
                out_img = visualize(img, boxes, classes)
                # plt.figure()
                # plt.imshow(out_img)
                # plt.show()
                name = str(i) + ".png"
                cv2.imwrite(str((out_dir / name).absolute()), out_img)

            frame_preds = {
                'Frame': i,
                'Classes': classes,  # we won't use them (yet)
                'BBoxes': boxes
            }

            
            # combine

            detections.append(frame_preds)
    else:
        img = np.array(frames)
        # OPTION 1. resize image
        target_w = 480
        target_h = 360
        img = cv2.resize(img, dsize=(target_w, target_h))
        # =============================

        boxes, classes, t_pred = predict(img, model, detection_threshold=THRESHOLD)
        pred_times.append(t_pred)

        #boxes, classes = keep_vehicles_only(boxes, classes)
        boxes, classes = keep_roadusers_only(boxes, classes)

        classes = classes.tolist()
        boxes = boxes.tolist()
        boxes, classes = combine_bboxes(boxes, classes)

        if show_predictions:
            out_img = visualize(img, boxes, classes)
            # plt.figure()
            # plt.imshow(out_img)
            # plt.show()
            name = str(i) + ".png"
            cv2.imwrite(str((out_dir / name).absolute()), out_img)

        frame_preds = {
            'Frame': 0,
            'Classes': classes,  # we won't use them (yet)
            'BBoxes': boxes
        }

        detections.append(frame_preds)

    avg_pred_time = sum(pred_times)/len(pred_times)
    print(f"\nAverage time to estimate detections for single frame: {avg_pred_time:.3f} & {1/avg_pred_time:.1f} FPS")
          
    return detections
    # json_save_path = Path(img_folder, f"frames.json")
    
    # if json_save_path.exists():
    #     with open(json_save_path, "r") as read_file:
    #         old_detections = json.load(read_file)
    #         detections = old_detections + detections
    # with open(json_save_path, "w") as write_file:
    #     json.dump(detections, write_file)
    #     print(f"Saved .json to {str(json_save_path)}")

