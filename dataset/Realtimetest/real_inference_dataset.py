import numpy as np
from PIL import Image
import PIL.Image
import cv2
from pathlib import Path


def get_masks(detections, img_size, method, prior_method, viz=False, divide=True):
    if not divide:
        print("WARNING: NOT DIVIDING BOX COORDINATES! Only allowed in visualization.")

    p_noise = 0.1
    thres_noise = int((1-p_noise)*100)

    if prior_method == 'car':
        prior = np.array(PIL.Image.open(
            Path("./priors/gta_prior_car_binary.jpg")))
        prior = denormalize(prior, 0, 1)
    elif prior_method == 'road':
        prior = np.array(PIL.Image.open(
            Path("./priors/gta_prior_road_binary.jpg")))
        prior = denormalize(prior, 0, 1)
    elif prior_method == 'union':
        prior1 = np.array(PIL.Image.open(
            Path("./priors/gta_prior_road_binary.jpg")))
        prior1 = denormalize(prior1, 0, 1)
        prior2 = np.array(PIL.Image.open(
            Path("./priors/gta_prior_car_binary.jpg")))
        prior2 = denormalize(prior2, 0, 1)
        prior = np.logical_or(prior1, prior2).astype(np.uint8)
    else:
        prior = np.ones((120, 160), dtype=np.uint8)

    if prior.shape != (120, 160):
        print(f"Resizing prior shape to fit image size of {img_size}")
        prior = cv2.resize(prior, (img_size[1], img_size[0]))

    assert method[:
                4] == 'case', f"Method must be case1, case2, case3, case4 but is {method}"
    case = int(method[-1])
    if case == 4:
        threshold = 4275
    else:
        threshold = 1728  # TRY 1728 == 1 % of 480x360 = 172800

    run_masks = np.zeros((1, 1, img_size[0], img_size[1]))
    frame_detections = detections[0]
    boxes = np.array(frame_detections['BBoxes'])
    classes = np.array(frame_detections['Classes'])
    processed_boxes, processed_classes = filter_boxes(
        boxes, classes, case, THRESHOLD=threshold)
    mask = masks_from_boxes(
        img_size, processed_boxes, divide_box_coordinates=divide)
    mask = mask * prior
    run_masks[0, 0, :, :] = mask
    if viz:
        return run_masks, processed_boxes, boxes # for debugging and visualization purposes
    else:
        return run_masks
    
def filter_boxes(boxes, classes, case, THRESHOLD=500):  # abblation study:
    num_boxes = boxes.shape[0]
    if num_boxes < 1:
        return boxes, classes
    if case == 1:
        # Case 1: do not filter anything, just process straight away
        return boxes, classes
    elif case == 2:
        # Case 2: only keep the biggest bounding box
        if num_boxes == 1:
            return boxes, classes
        else:
            largest_area = 0
            largest_bbox = np.zeros((1, 4))
            largest_class = 0
            for i in range(num_boxes):
                bbox = boxes[i, :]
                dx = bbox[2] - bbox[0]  # x2 - x1
                dy = bbox[3] - bbox[1]  # y2 - y1
                surface = dx*dy
                if surface > largest_area:
                    largest_area = surface
                    largest_bbox[0, :] = bbox
                    largest_class = classes[i]
            return largest_bbox, np.array([largest_class])
    elif case == 3 or case == 4:
        # Case 3: keep all bounding boxes larger than X
        boxes_keep = []
        classes_keep = []
        for i in range(num_boxes):
            bbox = boxes[i, :]
            dx = bbox[2] - bbox[0]  # x2 - x1
            dy = bbox[3] - bbox[1]  # y2 - y1
            surface = dx * dy
            # print(surface)
            if surface > THRESHOLD:
                boxes_keep.append(bbox)
                classes_keep.append(classes[i])
        return np.array(boxes_keep), np.array(classes_keep)
    else:
        return NotImplementedError("Only cases 1, 2, and 3 have been implemented")

def masks_from_boxes(img_size, boxes, divide_box_coordinates=True):
    """
    important note: the masks are calculated for the ORIGINAL size of the images, aka 360x480. Then resized to new size.
    """
    if divide_box_coordinates:
        h_new = img_size[0]
        w_new = img_size[1]
    else:
        h_new = 360
        w_new = 480
    mask = np.zeros((h_new, w_new))
    num_boxes = boxes.shape[0]
    if num_boxes == 0:
        mask = mask  # nothing changes
    else:
        for i in range(num_boxes):
            if divide_box_coordinates:
                bbox = boxes[i, :] / 3
            else:
                bbox = boxes[i, :]
            dx = bbox[2] - bbox[0]  # x2 - x1
            dy = bbox[3] - bbox[1]  # y2 - y1
            com = (bbox[0] + int(dx/2), bbox[1] + int(dy/2))
            radius = int((min(dx, dy) / 2))  # FOR CHANGING SIZES
            # radius = 5  # FOR FIXED SIZES
            object_mask = create_circular_mask(
                h_new, w_new, center=com, radius=int(radius))
            mask = np.logical_or(mask, object_mask)
    # resize the mask to new size
    mask = mask.astype(np.uint8)
    # mask = cv2.resize(mask, dsize=(w_new, h_new))
    return mask.astype(np.bool_)

def create_circular_mask(h, w, center, radius):
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask
    
def denormalize(img, min_sensor_val, max_sensor_val):
    """
    The data is provided in UINT8 values, however the actual sensor values are floats (and can also be negative).
    Therefore, we need to reverse the normalization process which was used to create the UINT8s. The normalization
    process was:
        UINT8 = ((sensor_reading - min_sensor_reading)*255) / (max_sensor_reading - min_sensor_reading)

    Therefore, the original sensor reading is recovered using:
        sensor_reading = (UINT8/255) * (max_sensor_reading - min_sensor_reading) + min_sensor_reading

    Args:
        img: image containing normalized UINT8 values
        min_sensor_val: minimum sensor value.
        max_sensor_val: maximum sensor value

    Returns:
        A denormalized matrix (same dimensions as img) of floating points.
    """
    max_sensor_val = float(max_sensor_val)
    min_sensor_val = float(min_sensor_val)
    out = (img / 255) * (max_sensor_val - min_sensor_val) + min_sensor_val
    return out