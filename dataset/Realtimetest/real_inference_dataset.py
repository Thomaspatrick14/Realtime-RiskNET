import numpy as np
from PIL import Image
import PIL.Image
import cv2
from pathlib import Path

class RealInferenceDataset:
    def __init__(self, detections, img_size, method, prior_method, viz=False, divide=True):
        self.detections = detections
        self.img_size = img_size
        self.method = method
        self.prior_method = prior_method
        self.viz = viz
        self.divide = divide
        if self.divide:
            self.h = self.img_size[0]
            self.w = self.img_size[1]
        else:
            self.h = 360
            self.w = 480

    def get_masks(self):
        if not self.divide:
            print("WARNING: NOT DIVIDING BOX COORDINATES! Only allowed in visualization.")

        p_noise = 0.1
        thres_noise = int((1 - p_noise) * 100)

        if self.prior_method == 'car':
            prior = np.array(PIL.Image.open(Path("./priors/gta_prior_car_binary.jpg")))
            prior = self.denormalize(prior, 0, 1)
        elif self.prior_method == 'road':
            prior = np.array(PIL.Image.open(Path("./priors/gta_prior_road_binary.jpg")))
            prior = self.denormalize(prior, 0, 1)
        elif self.prior_method == 'union':
            prior1 = np.array(PIL.Image.open(Path("./priors/gta_prior_road_binary.jpg")))
            prior1 = self.denormalize(prior1, 0, 1)
            prior2 = np.array(PIL.Image.open(Path("./priors/gta_prior_car_binary.jpg")))
            prior2 = self.denormalize(prior2, 0, 1)
            prior = np.logical_or(prior1, prior2).astype(np.uint8)
        else:
            prior = np.ones((120, 160), dtype=np.uint8)

        if prior.shape != (120, 160):
            print(f"Resizing prior shape to fit image size of {self.img_size}")
            prior = cv2.resize(prior, (self.img_size[1], self.img_size[0]))

        assert self.method[:4] == 'case', f"Method must be case1, case2, case3, case4 but is {self.method}"
        case = int(self.method[-1])
        if case == 4:
            threshold = 4275
        else:
            threshold = 1728  # TRY 1728 == 1 % of 480x360 = 172800

        run_masks = np.zeros((1, 1, self.img_size[0], self.img_size[1]))
        boxes = np.array(self.detections[0]['BBoxes'])
        mask, processed_boxes = self.filter_boxes(boxes, case, THRESHOLD=threshold)
        mask = mask * prior
        run_masks[0, 0, :, :] = mask
        if self.viz:
            return run_masks, processed_boxes, boxes  # for debugging and visualization purposes
        else:
            return run_masks

    def filter_boxes(self, boxes, case, THRESHOLD=500):  # abblation study:
        num_boxes = boxes.shape[0]
        mask = np.zeros((self.h, self.w))
        if num_boxes < 1:
            return mask.astype(np.bool_), np.array(boxes)
        
        if case == 3 or case == 4:
            # Case 3: keep all bounding boxes larger than X
            boxes_keep = []
            for i in range(num_boxes):
                bbox = boxes[i, :]
                x1, y1, x2, y2 = bbox
                dx = x2 - x1  # x2 - x1
                dy = y2 - y1  # y2 - y1
                surface = dx * dy
                if surface > THRESHOLD:
                    boxes_keep.append(bbox)
                    if self.divide:
                        x1 = x1 // 3
                        y1 = y1 // 3
                        x2 = x2 // 3
                        y2 = y2 // 3
                    com = (x1 + int(dx / 6), y1 + int(dy / 6))
                    radius = int((min(dx, dy) / 6))  # FOR CHANGING SIZES
                    object_mask = self.create_circular_mask(center=com, radius=int(radius))
                    mask = np.logical_or(mask, object_mask)
                mask = mask.astype(np.uint8)
            return mask.astype(np.bool_), np.array(boxes_keep)
        else:
            return NotImplementedError("Only cases 1, 2, and 3 have been implemented")

    def create_circular_mask(self, center, radius):
        Y, X = np.ogrid[:self.h, :self.w]
        dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

        mask = dist_from_center <= radius
        return mask

    def denormalize(self, img, min_sensor_val, max_sensor_val):
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