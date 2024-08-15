import torch
import numpy as np
import time
import cv2
import pycuda.driver as cuda
from tRTfiles.yolo_tensorrt_engine import do_inference

# class Detector:
#     def __init__(self, frame, context, tensorrt):

#         # target_classes = (0, 1, 2, 3, 5, 7)
#         self.combine_pairs = ((0, 1), (0, 3))
#         # combined_classes = ("cyclist", "motorcyclist")
#         self.combined_class_ids = (212, 214)
#         self.frames = frame
#         self.context = context
#         self.tensorrt = tensorrt

#     def combine_bboxes(self):
#         remaining_ids = list(range(len(self.boxes)))
#         new_bboxes = []
#         new_class_ids = []

#         for class_pair, new_class_id in zip(self.combine_pairs, self.combined_class_ids):
#             ids1 = [i for i, cid in enumerate(self.classes) if cid == class_pair[0]]
#             ids2 = [i for i, cid in enumerate(self.classes) if cid == class_pair[1]]
#             if len(ids1) == 0 or len(ids2) == 0:
#                 continue

#             iou_matrix = np.zeros((len(ids1), len(ids2)), dtype=float)
#             for i1, id1 in enumerate(ids1):
#                 for i2, id2 in enumerate(ids2):
#                     l1, t1, r1, b1 = self.boxes[id1] # l, t, r, b
#                     l2, t2, r2, b2 = self.boxes[id2]
#                     a1 = (r1-l1) * (b1-t1)
#                     a2 = (r2-l2) * (b2-t2)
#                     intersection = max(0, (min(r2, r1) - max(l1, l2))) * max(0, (min(b1, b2) - max(t1, t2)))
#                     iou_matrix[i1, i2] = float(intersection) / (a1 + a2)
#             ids1_for_ids2 = np.argmax(iou_matrix, axis=0)
#             matched_pairs = [(ids1[ci1], ids2[ci2]) for ci2, ci1 in enumerate(ids1_for_ids2) if iou_matrix[ci1, ci2] > 0.01]

#             for i1, i2 in matched_pairs:
#                 l1, t1, r1, b1 = self.boxes[i1]
#                 l2, t2, r2, b2 = self.boxes[i2]
#                 new_bbox = [min(l1, l2), min(t1, t2), max(r1, r2), max(b1, b2)]
#                 new_bboxes.append(new_bbox)
#                 new_class_ids.append(new_class_id)
#                 remaining_ids.remove(i1)
#                 remaining_ids.remove(i2)

#         if len(new_bboxes) > 0:
#             self.boxes = [bbox for i, bbox in enumerate(self.boxes) if i in remaining_ids]
#             self.classes = [class_id for i, class_id in enumerate(self.classes) if i in remaining_ids]
#             self.boxes.extend(new_bboxes)
#             self.classes.extend(new_class_ids)

#     def normalize_zero_one(self, data):
#         min_value = np.min(data)
#         max_value = np.max(data)
#         normalized = ((data - min_value) / (max_value - min_value))
#         return normalized

#     def keep_roadusers_only(self):
#         boxes_keep = []
#         classes_keep = []
#         keep_idx = [0, 1, 2, 3, 5, 7]
#         for i in range(self.classes.shape[0]):
#             if self.classes[i] in keep_idx:
#                 boxes_keep.append(self.boxes[i, :].tolist())
#                 classes_keep.append(self.classes[i])
#         self.boxes, self.classes = np.array(boxes_keep), np.array(classes_keep)

#     def do_inference(self, bindings, inputs, outputs, stream):
#         [cuda.memcpy_htod_async(inp['device'], inp['host'], stream) for inp in inputs]
#         self.context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
#         [cuda.memcpy_dtoh_async(out['host'], out['device'], stream) for out in outputs]
#         stream.synchronize()
#         return [out['host'] for out in outputs]

#     def predict(self):
#         detection_threshold = 0.9
#         inputs, outputs, bindings, stream = self.tensorrt
#         input_image = self.frames.transpose((2, 0, 1)).astype(np.float32)
#         input_image = self.normalize_zero_one(input_image)  # normalization doesnt seem to work, YOLO isn't detecting anything
#         # input_image /= 255.0
#         input_image = np.expand_dims(input_image, axis=0)

#         # Copy input image to pagelocked memory
#         np.copyto(inputs[0]['host'], input_image.ravel())

#         # Run inference
#         t_start = time.time()
#         trt_outputs = self.do_inference(bindings, inputs, outputs, stream)
#         t_pred = time.time() - t_start

#         # Process the output
#         trt_outputs = trt_outputs[0].reshape(1, 14175, 85)  # Adjust these dimensions based on your model output
#         pred_bboxes = trt_outputs[0][:, :4]
#         pred_scores = trt_outputs[0][:, 4]
#         pred_classes = trt_outputs[0][:, 5].astype(np.int32)

#         # Get boxes and classes above the threshold score
#         boxes = pred_bboxes[pred_scores >= detection_threshold].astype(np.int32)
#         classes = pred_classes[pred_scores >= detection_threshold]
#         print(f"Number of detections: {len(boxes)}")

#         # boxes[:, [1,3]] -= 60

#         return boxes, classes, t_pred

#     def detect(self):
#         detections = []
#         self.frames = cv2.copyMakeBorder(self.frames, 0, 120, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])

#         self.boxes, self.classes, t_pred = self.predict()

#         self.keep_roadusers_only()

#         self.classes = self.classes.tolist()
#         self.boxes = self.boxes.tolist()
#         self.combine_bboxes()

#         frame_preds = {
#             'Classes': self.classes,
#             'BBoxes': self.boxes
#         }

#         detections.append(frame_preds)
#         try:
#             print(f"\nTime to estimate detections for one frame: {t_pred:.3f} & {1/t_pred:.1f} FPS")
#         except ZeroDivisionError:
#             print(f"\nTime to estimate detections for one frame: {t_pred:.6f}")

#         return detections
    

target_classes = (0, 1, 2, 3, 5, 7)
combine_pairs = ((0, 1), (0, 3))
combined_classes = ("cyclist", "motorcyclist")
combined_class_ids = (212, 214)



def combine_bboxes(boxes, classes):
    if len(boxes) == 0 or len(classes) == 0:
        return boxes, classes

    remaining_ids = set(range(len(boxes)))
    new_boxes = []
    new_classes = []

    for class_pair, new_class_id in zip(combine_pairs, combined_class_ids):
        ids1 = [i for i in remaining_ids if classes[i] == class_pair[0]]
        ids2 = [i for i in remaining_ids if classes[i] == class_pair[1]]
        if not ids1 or not ids2:
            continue

        iou_matrix = np.zeros((len(ids1), len(ids2)), dtype=float)
        for i1, id1 in enumerate(ids1):
            for i2, id2 in enumerate(ids2):
                l1, t1, r1, b1 = boxes[id1]
                l2, t2, r2, b2 = boxes[id2]
                intersection = max(0, min(r1, r2) - max(l1, l2)) * max(0, min(b1, b2) - max(t1, t2))
                union = (r1-l1)*(b1-t1) + (r2-l2)*(b2-t2) - intersection
                iou_matrix[i1, i2] = intersection / union if union > 0 else 0

        ids1_for_ids2 = np.argmax(iou_matrix, axis=0)
        matched_pairs = [(ids1[ci1], ids2[ci2]) for ci2, ci1 in enumerate(ids1_for_ids2) if iou_matrix[ci1, ci2] > 0.01]

        for i1, i2 in matched_pairs:
            l1, t1, r1, b1 = boxes[i1]
            l2, t2, r2, b2 = boxes[i2]
            new_box = [min(l1, l2), min(t1, t2), max(r1, r2), max(b1, b2)]
            new_boxes.append(new_box)
            new_classes.append(new_class_id)
            remaining_ids.discard(i1)
            remaining_ids.discard(i2)

    # Add remaining boxes and classes
    final_boxes = [boxes[i] for i in remaining_ids] + new_boxes
    final_classes = [classes[i] for i in remaining_ids] + new_classes

    return final_boxes, final_classes

def keep_roadusers_only(boxes, classes):
    boxes_keep = []
    classes_keep = []
    keep_idx = [0, 1, 2, 3, 5, 7]
    for i in range(classes.shape[0]):
        if classes[i] in keep_idx:
            boxes_keep.append(boxes[i, :].tolist())
            classes_keep.append(classes[i])
    return np.array(boxes_keep), np.array(classes_keep)

#############################################################
####################### Not optimized #######################
#############################################################

# def predict(image, model, detection_threshold, cutoff_row=250):
#     # cutoff_row = int(image.shape[0] * 0.955)
#     # image[cutoff_row:, :, :] = 0 #if the bonnet is visible, it will be detected as a car
#     # image = normalize_zero_one(image) #normalization doesnt seem to work, YOLO isn't detecting anything

#     # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     # image = image.transpose((2, 0, 1))
#     # print(f"Shape of the image {image.shape}")
#     # image = torch.from_numpy(image).to(device)
#     # image = image.float().div(255.0).unsqueeze(0)

#     t_start = time.time()
#     with torch.no_grad():
#         output = model(image)
#     t_pred = time.time() - t_start
#     # print(f"Time taken: {t_pred}")
    
#     ## get all the predicited class names

#     # if image is loaded to cuda
#     # pred_bboxes = output[0][:, :4].detach().cpu().numpy()
#     # pred_scores = output[0][:, 4].detach().cpu().numpy()
#     # pred_classes = output[0][:, 5].detach().cpu().numpy().astype(int)

#     # if image is not loaded to cuda
#     pred_bboxes = output.pred[0][:, :4].detach().cpu().numpy()
#     pred_scores = output.pred[0][:, 4].detach().cpu().numpy()
#     pred_classes = output.pred[0][:, 5].detach().cpu().numpy().astype(int)

#     # get boxes and classes above the threshold score
#     boxes = pred_bboxes[pred_scores >= detection_threshold].astype(np.int32) # size of (x,4) where x is the number of detections and y is Xmin, Ymin, Xmax, Ymax
#     classes = pred_classes[pred_scores >= detection_threshold].astype(np.int32)

#     # Scale the bounding boxes from 640x480 to 480x360
#     # ratio = 360 / 480
#     # boxes[:, :] = boxes[:, :] * ratio
#     return boxes, classes, t_pred

# def detect(frames, model):
#     THRESHOLD = 0.5

#     detections = []

#     # =============================
#     # Resize the image to 640x480 (Yolo model expects dimensions which are a multiple of 32)
#     # width, height = 640, 480    # for cuda loaded image
#     # width, height = 480, 360
#     # img = cv2.resize(frames, (width, height), interpolation=cv2.INTER_LINEAR)
#     # =============================

#     boxes, classes, t_pred = predict(frames, model, detection_threshold=THRESHOLD)

#     #boxes, classes = keep_vehicles_only(boxes, classes)
#     boxes, classes = keep_roadusers_only(boxes, classes)

#     classes = classes.tolist()
#     boxes = boxes.tolist()
#     boxes, classes = combine_bboxes(boxes, classes)

#     frame_preds = {
#         'Frame': 0,
#         'Classes': classes,  # we won't use them (yet)
#         'BBoxes': boxes
#     }

#     detections.append(frame_preds)
#     try:
#         print(f"\nTime to estimate detections for one frame: {t_pred:.3f} & {1/t_pred:.1f} FPS")
#     except ZeroDivisionError:
#         print(f"\nTime to estimate detections for one frame: {t_pred:.6f}")
    
#     return detections

#############################################################
###################    TensorRT   ###########################
#############################################################

def predict(image, context, tensorrt, detection_threshold):

    # Preprocess the image
    image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB)
    input_image = image.transpose((2, 0, 1)).astype(np.float32)
    input_image /= 255.0
    input_image = np.expand_dims(input_image, axis=0)
    
    # Copy input image to pagelocked memory
    inputs, outputs, bindings, stream = tensorrt
    np.copyto(inputs[0]['host'], input_image.ravel())

    # Run inference
    t_start = time.time()
    trt_outputs = do_inference(context, bindings, inputs, outputs, stream)
    t_pred = time.time() - t_start

    # Process the output
    trt_outputs = trt_outputs[0].reshape(1, 300, 6)  # Adjust these dimensions based on your model output
    pred_bboxes = trt_outputs[0][:, :4]
    pred_scores = trt_outputs[0][:, 4]
    pred_classes = trt_outputs[0][:, 5].astype(np.int32)

    # Get boxes and classes above the threshold score
    boxes = pred_bboxes[pred_scores >= detection_threshold].astype(np.int32)
    classes = pred_classes[pred_scores >= detection_threshold]
    print(f"Number of detections: {len(boxes)}")

    return boxes, classes, t_pred

def detect(frames, context, tensorrt):
    THRESHOLD = 0.5

    detections = []
    img = cv2.copyMakeBorder(frames, 0, 120, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    boxes, classes, t_pred = predict(img, context, tensorrt, detection_threshold=THRESHOLD)

    boxes, classes = keep_roadusers_only(boxes, classes)

    classes = classes.tolist()
    boxes = boxes.tolist()
    boxes, classes = combine_bboxes(boxes, classes)

    frame_preds = {
        'Classes': classes,
        'BBoxes': boxes
    }

    detections.append(frame_preds)
    try:
        print(f"\nTime to estimate detections for one frame: {t_pred:.3f} & {1/t_pred:.1f} FPS")
    except ZeroDivisionError:
        print(f"\nTime to estimate detections for one frame: {t_pred:.6f}")
    
    return detections