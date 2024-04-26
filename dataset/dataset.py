import torch
from torch.utils.data import Dataset
import time
import numpy as np
from pathlib import Path
import json
from sys import platform
import os
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import scipy.ndimage
from mpl_toolkits.axes_grid1 import ImageGrid
import pickle
import PIL.Image
import random
import math


SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


class SiemensDataset(Dataset):
    """
    The main class for the Siemens synthetic dataset.
    """

    def __init__(self, mode, args, transform=None):
        img_size = [int(960 / args.downscale_factor),
                    int(1280 / args.downscale_factor)]
        if args.dataset in ['real_world', 'real_world_08', 'real_world_09']:
            real_world = True
        else:
            real_world = False

        self.transform = transform

        inputs = args.input
        h_flip = args.h_flip
        seq_len = args.T
        skip_factor = args.skip # JP: added new 31/10/2023
        safety_threshold = args.threshold # JP: added new 31/10/2023
        strat = args.sample_strat # JP: added new 05/12/2023
        if (mode == 'val' or mode == 'train') and strat == 'tim':
            skip_safe = 20  # was 20
            skip_act = 2  # was 2
            skip_urgent = 2  # was 2
            skip_exclude = 5
            if mode == 'val':
                h_flip = False
        
        exclude_frames = True # skips frames with labels -1

        # below is to align 30 FPS real world and 20 FPS sim datasets
        # try everything down to 1 second
        step = args.step
        assert step == 2, "Step needs to be 2 in order to hold to the 10 FPS input data requirement!"
        if real_world:
            """ Step needs to be 3, because the real-world data is 30 FPS. Training data is 20 FPS.
            Therefore, if we increase step from 2 to 3, we will still get 10 FPS data being fed in the network. 
            Following same reasoning, we also need to increase the sequence length 
            (24 frames at 30 FPS = 16 frames at 20 FPS) """
            print("Loading real-world data!")
            step = 1
            seq_len = int(args.T*0.5)

        
        if mode == 'test':
            h_flip = False
            skip_safe = 6
            skip_act = 1
            skip_urgent = 1
            skip_exclude = 1
            if real_world:
                skip_safe = 1
                # accounting for the 1.5 times FPS difference
                skip_safe = int(1.5*skip_safe)
                # accounting for the 1.5 times FPS difference
                skip_act = int(1.5*skip_act)
                # accounting for the 1.5 times FPS difference
                skip_urgent = int(1.5*skip_urgent)
                # accounting for the 1.5 times FPS difference
                skip_exclude = int(1.5*skip_exclude)

        if args.tiny_dataset:
            # only use a few (specified) runs in order to quickly debug
            # --> 20230926: small dataset only for debugging (only 4 instead of 200 runs for quick testing)
            RUNS = get_tiny_dataset(mode)
            # RUNS = exclude_runs(RUNS)
        elif args.single_run >= 0:
            # only return the data for one single run. Used mainly in evaluation
            print(f"Only creating data for run {args.single_run}")
            RUNS = [args.single_run]
        elif real_world:
            # Samples where we have GT annotations: --> 20230926: Tim removed some runs because they were not labeled correctly (GT)
            RUNS = [1, 2, 3, 4, 5, 7, 9, 10, 11, 12, 16, 21, 14, 15, 18, 19]
        else:
            # # normal mode: get the runs based on run mode (train/val/test) and the K-fold 
            # if real_world:  # TODO: this will never get activated right? coz there is a elif real_world right above this
            #     raise UserWarning(
            #         "Trying to load a k-fold from the real-world dataset, where k-folds are not created!")
            # else:
            RUNS = get_k_fold(mode, args)

            # RUNS = exclude_runs(RUNS)

        if platform == "win32":
            result_path = Path(
                "C:/Users/up650/Downloads/constant_radius", args.dataset, "Runs")
        else:
            result_path = Path("/storage/users/j.p.udhayakumar/datasets", args.dataset, "Runs")

        # sorted ensures compatibility between linux OS
        all_results = sorted(os.listdir(result_path))

        # initialization
        first_data = True
        n_runs = len(RUNS)
        counter = 0
        t_start = time.time()
        for RUN in RUNS:
            counter += 1
            t_run_start = time.time()
            run_folder = result_path / Path(all_results[RUN])

            run_labels = get_labels(run_folder)
            if run_labels is None:
                print(f"Labels for run {RUN} do not exist!")
                continue
            sequence_info = get_sequence_info(run_labels, seq_len, skip_safe, skip_act, skip_urgent, skip_exclude, exclude_frames, skip_factor, safety_threshold, strat)
            run_labels = get_data_as_sequences(
                run_labels, sequence_info, step, label_mode=True, hflip=h_flip)

            if 'flow' in inputs:
                run_flow = get_flow(run_folder, img_size, real_world)
                run_flow = get_data_as_sequences(
                    run_flow, sequence_info, step, hflip=h_flip)
            else:
                run_flow = np.array([0])

            if 'depth' in inputs:
                run_depth = get_depth(run_folder, img_size, real_world)
                run_depth = get_data_as_sequences(
                    run_depth, sequence_info, step, hflip=h_flip)
            else:
                run_depth = np.array([0])

            if 'rgb' in inputs:
                run_rgb = get_rgb(run_folder, img_size, real_world)
                run_rgb = get_data_as_sequences(
                    run_rgb, sequence_info, step, hflip=h_flip)
            else:
                run_rgb = np.array([0])

            if 'mask' in inputs:
                run_masks = get_masks(
                    run_folder, img_size, args.mask_method, real_world, args.mask_prior, mode)
                run_masks = get_data_as_sequences(
                    run_masks, sequence_info, step, hflip=h_flip)
                # run_masks = np.array([0])
            else:
                run_masks = np.array([0])

            if first_data:  # 20230926: first time you can't stack on NaNs, so this first time.
                self.labels = run_labels
                self.flow = run_flow
                self.depth = run_depth
                self.rgb = run_rgb
                self.masks = run_masks
                first_data = False
            else:
                self.labels = np.hstack((self.labels, run_labels))
                self.flow = np.vstack((self.flow, run_flow))
                self.depth = np.vstack((self.depth, run_depth))
                self.rgb = np.vstack((self.rgb, run_rgb))
                self.masks = np.vstack((self.masks, run_masks))

            print(f"Run: {RUN} ({counter}/{n_runs}) \t Loaded in: {(time.time() - t_run_start):.0f} s \t Folder: "
                  f"{run_folder}")

        self.labels = np.squeeze(self.labels)
        self.n_samples = self.labels.size
        print(f"Loaded data in {(time.time() - t_start)/60:.1f} minutes")
        self.unique, self.counts = np.unique(self.labels, return_counts=True)
        print(f"Unique labels: {self.unique}. Counts: {self.counts}")

        if args.visualize:
            if len(self.unique) == 3:
                plt.figure()
                plt.bar(self.unique[0], self.counts[0],
                        color='tab:green', width=0.9)
                plt.bar(self.unique[1], self.counts[1],
                        color='tab:orange', width=0.9)
                plt.bar(self.unique[2], self.counts[2],
                        color='tab:red', width=0.9)
                plt.xlabel("Label")
                plt.ylabel("# of sequences")
                plt.grid()
                plt.show()
            elif len(self.unique) == 2:
                font = {'family': 'normal',
                        'weight': 'normal',
                        'size': 16}

                plt.rc('font', **font)
                plt.figure()
                plt.bar(self.unique[0], self.counts[0],
                        color='tab:green', width=0.9)
                plt.bar(self.unique[1], self.counts[1],
                        color='tab:red', width=0.9)
                plt.xlabel("Label")
                plt.xticks([0, 1])
                plt.ylabel("# of sequences")
                plt.grid()
                plt.show()

            visualize_risk(self.flow, self.depth, self.rgb,
                           self.masks, self.labels, step, threshold=0.999)

    def __getitem__(self, index):
        label = self.labels[index]

        if len(self.flow.shape) > 3:
            flow = self.flow[index]
        else:
            flow = np.zeros_like(np.array([label]))

        if len(self.depth.shape) > 3:
            depth = self.depth[index]
        else:
            depth = np.zeros_like(np.array([label]))

        if len(self.rgb.shape) > 3:
            rgb = self.rgb[index]
        else:
            rgb = np.zeros_like(np.array([label]))

        if len(self.masks.shape) > 3:
            mask = self.masks[index]
        else:
            mask = np.zeros_like(np.array([label]))

        if self.transform:
            mask = self.transform(mask)
            mask = self.transform(mask)
            mask = self.transform(mask)

        flow = torch.Tensor(flow)
        depth = torch.Tensor(depth)
        rgb = torch.Tensor(rgb)
        mask = torch.Tensor(mask)
        # --> 20230926: labels is a numpy array, so not a tensor. Depth would be an easy extension to add with a Transformer
        return flow, depth, rgb, mask, label

    def __len__(self):
        return self.n_samples


class MaskNoise(object):
    def __init__(self, p_noise=25):
        self.p_noise = p_noise
        self.x = [0, 160]
        self.y = [50, 100]
        self.h = 120
        self.w = 160

    def __call__(self, mask):
        rand = np.random.randint(0, 100)
        if rand < self.p_noise:
            radius = np.random.randint(5, 20)
            x = np.random.randint(self.x[0] + radius, self.x[1] - radius)
            y = np.random.randint(self.y[0], self.y[1])
            x_vars = np.random.randint(-3, 3, mask.shape[1]) #JP: will work for any T (history time)
            y_vars = np.random.randint(-3, 3, mask.shape[1])
            r_vars = np.random.randint(-3, 3, mask.shape[1])
            noise_mask = np.zeros_like(mask)
            for i, (dx, dy, dr) in enumerate(zip(x_vars, y_vars, r_vars)):
                pos = (x + dx, y + dy)
                r = radius + dr
                noise_mask[0, i, :, :] = create_circular_mask(
                    self.h, self.w, pos, r).astype(np.int)
            noise_mask[0, :, :, :] = np.logical_or(mask, noise_mask)
            return noise_mask
        else:
            return mask


def get_labels(run_folder):
    labels_path = run_folder / 'labels.csv'

    exists = os.path.exists(labels_path)
    if exists:
        run_labels_data = np.loadtxt(labels_path, delimiter=',')
        return run_labels_data
    else:
        return None


def get_flow(run_folder, img_size, real):
    if real:
        flow_vx_sensor = 'fx'
        flow_vy_sensor = 'fy'
    else:
        if platform == "win32":  # TODO: this is a temporary solution! fix this by getting liteflownet pred on windows
            flow_vx_sensor = "PhysicsBasedCameraUnreal_1_OpticalFlow_X"  # perfect OF
            flow_vy_sensor = "PhysicsBasedCameraUnreal_1_OpticalFlow_Y"  # perfect OF
        else:
            flow_vx_sensor = "fx"  # training on liteflownet predictions on synthetic data
            flow_vy_sensor = "fy"  # training on liteflownet predictions on synthetic data
            # flow_vx_sensor = "fx_processing4"
            # flow_vy_sensor = "fy_processing4"

    # flow_parameters_path = run_folder / 'OpticalMinMax.json'
    # if flow_parameters_path.exists():
    #     with open(flow_parameters_path) as json_file:
    #         parameters = json.load(json_file)
    # else:
    #     flow_parameters_path = run_folder / "PhysicsBasedCameraUnreal_1_OpticalFlow" / 'OpticalMinMax.json'
    #     with open(flow_parameters_path) as json_file:
    #         parameters = json.load(json_file)

    # sorted ensures compatibility between OS
    vx_imgs = sorted(os.listdir(run_folder / flow_vx_sensor))
    # sorted ensures compatibility between OS
    vy_imgs = sorted(os.listdir(run_folder / flow_vy_sensor))

    num_steps = len(vx_imgs)
    run_flow = np.zeros(
        (2, num_steps, img_size[0], img_size[1]), dtype=np.float32)

    for step in range(num_steps):
        vx_path = run_folder / flow_vx_sensor / vx_imgs[step]
        vx = np.array(Image.open(vx_path))
        if len(vx.shape) == 3:
            vx = vx[:, :, 0]
        # vx_min = parameters[step]['MinX']
        # vx_max = parameters[step]['MaxX']
        # de-normalization
        # vx = denormalize(vx, vx_min, vx_max) * flow_factor
        # vx = vx/abs(vx).max()
        run_flow[0, step, :, :] = vx / 255.

        vy_path = run_folder / flow_vy_sensor / vy_imgs[step]
        vy = np.array(Image.open(vy_path))
        if len(vy.shape) == 3:
            vy = vy[:, :, 0]
        # vy_min = parameters[step]['MinY']
        # vy_max = parameters[step]['MaxY']
        # vy = denormalize(vy, vy_min, vy_max) * flow_factor
        # vy = vy / abs(vy).max()
        run_flow[1, step, :, :] = vy / 255.
    return run_flow


def get_depth(run_folder, img_size, real):
    if real:
        raise ValueError("Depth data is not available for real-world dataset!")
    depth_sensor = "PhysicsBasedCameraUnreal_1_Distance"
    # sorted ensures compatibility between OS
    depth_imgs = sorted(os.listdir(run_folder / depth_sensor))

    num_steps = len(depth_imgs)
    run_depths = np.zeros(
        (1, num_steps, img_size[0], img_size[1]), dtype=np.float32)

    for step in range(num_steps):
        depth_path = run_folder / depth_sensor / depth_imgs[step]
        img = np.array(Image.open(depth_path))[:, :, 0]
        # de-normalizing
        img = denormalize(img, min_sensor_val=0, max_sensor_val=65)
        run_depths[0, step, :, :] = img
    return run_depths


def get_rgb(run_folder, img_size, real):
    if real:
        rgb_sensor = "rgb"
    else:
        rgb_sensor = "CameraSensor_1"

    # # TODO: Only for testing purposes this is set to HSV, aka flow!
    # rgb_sensor = "DarkFlow"

    if not Path(run_folder / rgb_sensor).exists():
        rgb_sensor = "PhysicsBasedCameraUnreal_1_ImageRGBOutput"
    #     rgb_imgs = sorted(os.listdir(run_folder / rgb_sensor))  # sorted ensures compatibility between OS
    # else:
    #     rgb_sensor = "PhysicsBasedCameraUnreal_1_ImageRGBOutput"
    #     rgb_imgs = sorted(os.listdir(run_folder / rgb_sensor))  # sorted ensures compatibility between OS

    # sorted ensures compatibility between OS
    rgb_imgs = sorted(os.listdir(run_folder / rgb_sensor))

    num_steps = len(rgb_imgs)
    run_rgb = np.zeros(
        (3, num_steps, img_size[0], img_size[1]), dtype=np.float32)

    for step in range(num_steps):
        rgb_path = run_folder / rgb_sensor / rgb_imgs[step]
        img = np.array(Image.open(rgb_path))
        # transform WxHx3 to 3xWxH --> # TorchVision models verwachten TxCxWxH: time, channel, width, height
        rgb = np.moveaxis(img, -1, 0)
        run_rgb[:, step, :, :] = rgb
    return run_rgb


def get_masks(run_folder, img_size, method, real, prior_method, mode, divide=True):
    if not divide:
        print("WARNING: NOT DIVIDING BOX COORDINATES! Only allowed in visualization.")

    p_noise = 0.1
    thres_noise = int((1-p_noise)*100)
    n_noise_sequences = 6

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

    if prior.shape != (img_size[0], img_size[1]):
        print(f"Resizing prior shape to fit image size of {img_size}")
        prior = cv2.resize(prior, (img_size[1], img_size[0]))

    if real:
        assert method[:
                      4] == 'case', f"Method must be case1, case2, case3, case4 but is {method}"
        case = int(method[-1])
        if case == 4:
            threshold = 4275
        else:
            threshold = 1728  # TRY 1728 == 1 % of 480x360 = 172800

        json_path = Path(run_folder, "detections.json")
        with open(json_path) as f:
            detections = json.load(f)

        num_steps = len(detections)
        run_masks = np.zeros((1, num_steps, img_size[0], img_size[1]))
        for step in range(num_steps):
            frame_detections = detections[step]
            boxes = np.array(frame_detections['BBoxes'])
            classes = np.array(frame_detections['Classes'])
            processed_boxes, processed_classes = filter_boxes(
                boxes, classes, case, THRESHOLD=threshold)
            mask = masks_from_boxes(
                img_size, processed_boxes, divide_box_coordinates=divide)
            mask = mask * prior
            run_masks[0, step, :, :] = mask
        # plot_masks(run_masks[0, :, :, :])
        return run_masks
    else:
        label_sensor = "PhysicsBasedCameraUnreal_1_ImageSegmentation"
        # sorted ensures compatibility between OS
        seg_imgs = sorted(os.listdir(run_folder / label_sensor))
        if 'CaptureInfo.xml' in seg_imgs:
            seg_imgs.remove('CaptureInfo.xml')
        num_steps = len(seg_imgs)
        run_masks = np.zeros((1, num_steps, img_size[0], img_size[1]))
        allowed_extensions = ['.jpg']
        
        for step in range(num_steps):
            depth_path = run_folder / label_sensor / seg_imgs[step]
            if os.path.isfile(depth_path) and depth_path.suffix.lower() in allowed_extensions:
                img = np.array(Image.open(depth_path))
                labels = create_annotations(img)
                mask = get_region_mask(labels, method=method)
                mask = mask * prior
                if thres_noise > 0:
                    rand = random.randint(0, 100)
                    if rand > thres_noise and mode == 'train':
                        mask = add_noise_to_mask(mask)
                run_masks[0, step, :, :] = mask

        # if mode == 'train':
        #     for i in range(n_noise_sequences):
        #         run_masks[0, :, :, :] = add_mask_sequence_noise(run_masks[0, :, :, :], max_duration=50)
        # plot_masks(run_masks[0, :, :, :])
        return run_masks


def plot_masks(masks):
    n_masks = masks.shape[0]
    white_bar = np.ones((120, 3))
    r1_end = n_masks - 1
    r2_end = n_masks - int(n_masks / 3)
    r3_end = n_masks - int(2*n_masks / 3)

    for i in range(10):
        if i == 0:
            r1 = masks[r1_end, :, :]
            r2 = masks[r2_end, :, :]
            r3 = masks[r3_end, :, :]
        else:
            r1 = np.concatenate(
                (r1, white_bar, masks[r1_end-int(i*2), :, :]), axis=1)
            r2 = np.concatenate(
                (r2, white_bar, masks[r2_end-int(i*2), :, :]), axis=1)
            r3 = np.concatenate(
                (r3, white_bar, masks[r3_end-int(i*2), :, :]), axis=1)
    width = r1.shape[1]
    white_line = np.ones((3, width))
    plot = np.concatenate((r1, white_line, r2, white_line, r3), axis=0)

    plt.imshow(plot)
    plt.show()


def add_noise_to_mask(mask, radius=14):
    # requires mask to be binary, 0 to 1.
    # radius 14 = 3% of image
    x = random.randint(0, mask.shape[1])
    y = random.randint(30, 70)
    radius = random.randint(5, 20)  # FOR CHANGING MASK RADIUS
    # radius = 5  # FOR FIXED MASK RADIUS
    pos = (x, y)
    noise_mask = create_circular_mask(
        mask.shape[0], mask.shape[1], pos, radius).astype(np.int)
    mask = np.logical_or(mask, noise_mask)
    return mask


def add_mask_sequence_noise(masks, min_duration=0, max_duration=50):
    mask_shape = [masks.shape[1], masks.shape[2]]
    n_masks = masks.shape[0]
    x = [0, 160]
    y = [50, 100]

    radius = random.randint(5, 20)  # FOR FIXED RADIUS
    # radius = 5  # FOR CHANGING RADIUS
    f_start = random.randint(0, n_masks)
    duration = random.randint(min_duration, max_duration)
    if f_start + duration >= n_masks:
        f_end = n_masks
    else:
        f_end = f_start + duration

    x = random.randint(x[0] + radius, x[1] - radius)
    y = random.randint(y[0], y[1])
    pos = (x, y)
    noise_mask = create_circular_mask(
        mask_shape[0], mask_shape[1], pos, radius).astype(np.int)
    masks[f_start:f_end] = np.logical_or(masks[f_start:f_end], noise_mask)
    # print(f"Radius: {radius}. F_start: {f_start}. F_end: {f_end}. X: {x}, Y: {y}")
    return masks


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


# masks are circles, so to go from rectangles (BBs) to circles, heuristics are applied here
def masks_from_boxes(img_size, boxes, divide_box_coordinates=True):
    """
    important note: the masks are calculated for the ORIGINAL size of the images, aka 360x480. Then resized to new size.
    """
    h_new = img_size[0]
    w_new = img_size[1]
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
    return mask.astype(np.bool)


def get_data_as_sequences(data, sequence_info, step=2, label_mode=False, hflip=False):
    output = np.array([])
    first_data = True
    for info in sequence_info:
        f_start = info['f_start']
        f_end = info['f_end']
        label = info['label']
        if label_mode:
            if hflip and (label > 0):
                sequence = np.hstack((label, label))
            else:
                sequence = label
                # because we stack in over a new axis
                sequence = np.expand_dims(sequence, axis=0)
        else:
            # CRUCIAL: GRAB FROM f_start + STEP to avoid lag!!
            sequence = data[:, f_start+step:f_end+step:step, :, :]
        # because we stack in over a new axis
        sequence = np.expand_dims(sequence, axis=0)
        if hflip and (label > 0) and (label_mode == False):
            sequence_flipped = h_flip_sequence(sequence)
            sequence = np.vstack((sequence, sequence_flipped))

        if first_data:
            output = sequence
            first_data = False
        else:
            if label_mode:
                output = np.hstack((output, sequence))
            else:
                output = np.vstack((output, sequence))
    return output


# 20230926: Tim says: not very neat
def get_sequence_info(labels, seq_len, skip_safe, skip_act, skip_urgent, skip_exclude, exclude_frames, skip_factor, safety_threshold, strat):
    sequence_info = []
    labels = np.array(labels)
    if exclude_frames:
        mock = labels[labels != -1].copy()
    else:
        mock = labels.copy() 

    f_start = 0
    f_end = seq_len - 1
    
    mock[mock == 1] = 0
    mock[mock == 2] = 1

    while f_end < (mock.shape[0] - 1):
        occ = np.count_nonzero(mock[f_start : f_end + 1] == 1)

        if strat == ('tim' or 'fair'):
            if mock[f_end]==1:
                label = 1
            else:
                label = 0
        elif strat == 'jp':
            if occ>(safety_threshold*seq_len) or mock[f_end]==1: # JP: the sequence is considered to be unsafe if more than 30% of the frames in the sequence are unsafe
                label = 1
            else:
                label = 0
        else:
            raise UserWarning("Mentioned Sampling Strategy does not exist")

        info = {
                    'f_end': f_end,
                    'f_start': f_start,
                    'label': label
                } 
        sequence_info.append(info)

        if strat == 'tim':
            if label == 0:
                f_start += skip_safe
                f_end += skip_safe
            elif label == 1:
                f_start += skip_urgent
                f_end += skip_urgent
            else:
                raise ValueError
            
        elif strat == 'jp': 
            if mock[f_end] == 1: # JP's seq strat
                f_start += math.ceil(seq_len*skip_factor)
                f_end += math.ceil(seq_len*skip_factor)
            else:
                f_start += seq_len
                f_end += seq_len

        elif strat == 'fair':
            f_start += 2 #JP: Fair seq
            f_end += 2
        else:
            raise UserWarning("Mentioned Sampling Strategy does not exist")

    return sequence_info


def get_tiny_dataset(mode):
    """
    Returns a specific and very short list of runs. Useful for debugging and quick checks.

    Args:
        mode: string, 'train', 'val, 'test'. Indicates purpose of dataset (training, validation, testing).

    Returns:
        List of runs for the given mode
    """
    if mode == 'train':
        return [i for i in range(5, 9)]  # [i for i in range(0, 2)]
    elif mode == 'val':
        return [4]  # [i for i in range(0, 10)]  # [i for i in range(100, 102)]
    elif mode == 'test':
        return [5]  # [i for i in range(200, 202)]
    else:
        raise ValueError(
            f"Mode {mode} is not specified. Please use train, val, or test.")


def get_k_fold(mode, args):
    """
    Returns a list of runs belonging to a certain fold of the dataset. The entire dataset is divided into 5 folds,
    3 folds are used for training, 1 for validation, 1 for test.

    Note: requires a file 'k_fold.csv' to be present! This file contains all run numbers, randomly shuffled. This must
    remain constant between different folds for fair evaluation, and therefore needs to be loaded in.

    Args:
        mode: string, 'train', 'val, 'test'. Indicates purpose of dataset (training, validation, testing).
        args: Namespace with the arguments to be used in the model selection. The required argument is k_fold.
              k_fold: int, from 0 up to and including 4.

    Returns:
        A list of runs belonging to this certain fold of the dataset and the given mode.
    """

    k = args.k_fold
    assert k < 5, "Maximum value of K is 4 (5-folding)"

    training_runs = []
    validation_runs = []
    test_runs = [] # JP: test is on real world dataset, so test_runs will never be populated

    if args.dataset == "extended":
        print(f"Loading the extended dataset k_folds!")
        runs = np.loadtxt('/storage/users/j.p.udhayakumar/jobs/RiskNET/dataset/extended_kfold.csv', delimiter=',')
        runs = np.array(runs, dtype=int)
        fold0 = list(runs[0:56])
        fold1 = list(runs[56:112])
        fold2 = list(runs[112:168])
        fold3 = list(runs[168:224])
        fold4 = list(runs[224:])

    elif args.dataset == "constant_radius":
        print(f"Loading the constant_radius dataset k_folds!")
        runs = np.loadtxt('/storage/users/j.p.udhayakumar/jobs/RiskNET/dataset/constant_radius_kfold.csv', delimiter=',')
        runs = np.array(runs, dtype=int)
        fold0 = list(runs[0:38])
        fold1 = list(runs[38:76])
        fold2 = list(runs[76:114])
        fold3 = list(runs[114:152])
        fold4 = list(runs[152:])
        
    elif args.dataset == "constant_straight":
        print(f"Loading the constant_straight dataset k_folds!")
        runs = np.loadtxt('/storage/users/j.p.udhayakumar/jobs/RiskNET/dataset/constant_straight_kfold.csv', delimiter=',')
        runs = np.array(runs, dtype=int)
        div = 18
        fold0 = list(runs[0:div * 1])
        fold1 = list(runs[div * 1:div * 2])
        fold2 = list(runs[div * 2:div * 3])
        fold3 = list(runs[div * 3:div * 4])
        fold4 = list(runs[div * 4:])
    else:
        runs = np.loadtxt('/storage/users/j.p.udhayakumar/jobs/RiskNET/dataset/k_fold.csv', delimiter=',')
        runs = np.array(runs, dtype=int)
        fold0 = list(runs[0:60])
        fold1 = list(runs[60:119])
        fold2 = list(runs[119:178])
        fold3 = list(runs[178:237])
        fold4 = list(runs[237:297])

    if k == 0:
        training_runs.extend(fold0)
        training_runs.extend(fold1)
        training_runs.extend(fold2)
        training_runs.extend(fold3)
        validation_runs.extend(fold4) # TODO: testing out variations
    elif k == 1:
        training_runs.extend(fold4)
        training_runs.extend(fold0)
        training_runs.extend(fold1)
        training_runs.extend(fold2)
        validation_runs.extend(fold3)
    elif k == 2:
        training_runs.extend(fold3)
        training_runs.extend(fold4)
        training_runs.extend(fold0)
        training_runs.extend(fold1)
        validation_runs.extend(fold2)
    elif k == 3:
        training_runs.extend(fold2)
        training_runs.extend(fold3)
        training_runs.extend(fold4)
        training_runs.extend(fold0)
        validation_runs.extend(fold1)
    elif k == 4:
        training_runs.extend(fold1)
        training_runs.extend(fold2)
        training_runs.extend(fold3)
        training_runs.extend(fold4)
        # Tim : actually not a test set, but a validation set, since testing is not done on SIM data, but on real world data
        validation_runs.extend(fold0)
    elif k == 10:
        # CUSTOM K
        training_runs.extend(fold1)
        test_runs.extend(fold2)
    if mode == 'train':
        return training_runs
    elif mode == 'val':
        return validation_runs
    elif mode == 'test' or mode == 'final_test':
        return validation_runs


def exclude_runs(runs):
    """
    Excludes certain runs from the dataset. There are multiple reasons for this:
    1) Collisions out of the FoV of the ego agent, such as rear-ends and T-bones.
    2) Collisions are with a bus. Busses are currently not represented in flow and depth, therefore lead to wrong labels --> 20230926: this applies to the PreScan data where OF was extracted from the physics models (not from RGB data)
    3) Safe scenarios. Using the passed arguments, one can chose to exclude all safe driving scenarios --> 20230926: There are safe scenarios, but Tim has filtered out especially the videos in which there are no unsafe scenarios at all
    Args:
        runs: List of all runs which will be used in the dataset (if nothing would be excluded)
        args: Namespace with the arguments to be used in the model selection. The required arguments:
              skip_safe_runs: if True, excludes safe driving scenarios. If False, does not exclude safe driving.

    Returns:
        A new list of all runs, excluding frames for reasons 1) and 2) and based on settings 3).
    """
    excluded = []
    non_collisions = [3, 4, 11, 12, 13, 14, 17, 18, 19, 20, 23, 24, 28, 31, 32, 33, 39, 47, 49, 52, 54, 55, 58, 61, 66,
                      71, 72, 74, 77, 82, 86, 93, 97, 98, 101, 102, 107, 110, 113, 114, 121, 127, 131, 136, 139, 141,
                      143, 144, 149, 155, 157, 164, 168, 171, 173, 178, 189, 190, 193, 197, 202, 205, 206, 208, 215,
                      216, 221, 224, 232, 236, 243, 245, 250, 255, 266, 269, 279, 280, 281, 284, 285, 287, 288, 295]
    out_of_FoV = [0, 2, 7, 9, 10, 15, 21, 25, 26, 30, 35, 38, 41, 42, 45, 48, 51, 59, 63, 64, 65, 68, 69, 70, 73, 78,
                  79, 85, 87, 89, 90, 100, 104, 106, 108, 112, 116, 118, 120, 122, 128, 134, 135, 138, 140, 142, 145,
                  152, 153, 154, 160, 161, 162, 163, 165, 167, 169, 170, 174, 175, 177, 180, 182, 183, 186, 187, 192,
                  194, 195, 198, 199, 200, 201, 203, 207, 210, 212, 213, 214, 217, 218, 219, 222, 223, 225, 227, 229,
                  230, 231, 233, 234, 235, 237, 239, 240, 241, 242, 244, 246, 247, 257, 263, 267, 270, 273, 275, 277,
                  278, 282, 283, 289, 290, 292, 293, 294]

    bus_collisions = [5, 21, 34, 40, 50, 63, 67, 69, 92, 94, 104, 109, 123, 147, 148, 151, 158, 172, 180, 203, 204, 212,
                      217, 222, 223, 226, 230, 240, 241, 249, 252, 258, 271, 282, 291]

    excluded.extend(out_of_FoV)
    excluded.extend(bus_collisions)

    filtered_runs = []
    skipped = 0
    for run in runs:
        if run not in excluded:
            filtered_runs.append(run)
        else:
            skipped += 1
    print("Excluded {} runs, kept {} runs".format(skipped, len(filtered_runs)))
    return filtered_runs


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


def create_annotations(img):    # PreScan does not supply BBs, only SS. This creates BBs filtered from SS maps. Since pixel values ​​are not completely the same, you have to use a range here (e.g. green SS pixel of truck on the front, is a different type of green than SS pixel on the back)
    car_class = 1
    car_upper = np.array([30, 30, 240])
    car_lower = np.array([0, 0, 180])
    truck_class = 2
    truck_upper = np.array([30, 256, 30])
    truck_lower = np.array([0, 200, 0])
    bike_class = 3
    bike_upper = np.array([256, 30, 30])
    bike_lower = np.array([200, 0, 0])

    labels = np.zeros((img.shape[0], img.shape[1]), dtype=int)

    car_mask = np.all((img <= car_upper) & (img >= car_lower), axis=-1)
    truck_mask = np.all((img <= truck_upper) & (img >= truck_lower), axis=-1)
    bike_mask = np.all((img <= bike_upper) & (img >= bike_lower), axis=-1)

    labels[car_mask] = car_class
    labels[truck_mask] = truck_class
    labels[bike_mask] = bike_class

    # always filter out ego hood
    labels[107:, :] = 0
    return labels


def get_region_mask(labels, method='gaussian', th=25, size_factor=1):
    h = labels.shape[0]
    w = labels.shape[1]

    # find objects in labels
    individually_labeled, n_objects = scipy.ndimage.label(labels)
    index = [i for i in range(1, n_objects + 1)]
    centers_of_mass = scipy.ndimage.center_of_mass(
        labels, individually_labeled, index)
    sum_of_pixels = scipy.ndimage.sum(labels, individually_labeled, index)

    # filter objects:
    filtered_objects = []
    for i, (p_sum, com) in enumerate(zip(sum_of_pixels, centers_of_mass)):
        if p_sum >= th:  # Remove if less than TH pixels
            # FOR CHANGING RADIUS
            radius = int(size_factor * (np.sqrt(p_sum) / np.pi))
            # radius = 5  # FOR FIXED RADIUS
            # Creates a list with (row_CoM, column_CoM), sum_of_pixels
            object = [(com[1], com[0]), radius, p_sum]
            filtered_objects.append(object)

    # creating a mask which can be put over the labels --> 20230926: Tim did some testing, but Gaussian or Binary made no difference
    if method == 'binary':
        mask = np.zeros((h, w), dtype=bool)
        for i, (com, radius, p_sum) in enumerate(filtered_objects):
            object_mask = create_circular_mask(h, w, center=com, radius=radius)
            mask = np.logical_or(mask, object_mask)
    elif method == 'gaussian':
        mask = np.zeros((h, w), dtype=np.float32)
        for i, (com, radius, p_sum) in enumerate(filtered_objects):
            object_mask = make_gaussian(com, fwhm=radius)
            mask += object_mask
    else:
        mask = labels > 0
    return mask


def make_gaussian(center_of_mass, fwhm=3, h=120, w=160):
    x0 = center_of_mass[0]
    y0 = center_of_mass[1]

    x = np.arange(0, w, 1, float)
    y = x[:, np.newaxis]
    gauss = np.exp(-4 * np.log(2) * ((x - x0) **
                   2 + (y - y0) ** 2) / fwhm ** 2)
    return gauss[:h, :]


def create_circular_mask(h, w, center, radius):
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask


def h_flip_sequence(sequence):
    """
    Horizontally flips all data channels in the provided sequence and stores this by adding to the first dimension.
    X flow are multiplied by -1 to keep the direction of the flow correct. Y flow stays the same, because the direction is the same in both cases.

    Args:
        sequence: a 5 dimensional matrix. 1st dimension is 1, second dimension is either (depth), (vx, vy) or
                  (red, green, blue). Third dimension is the length of the sequence, fourth and fifth are the size of
                  the images.

    Returns:
        Sequence with one extra value in the first dimension, namely the newly created horizontally flipped data.
    """
    flow = False
    layers = sequence.shape[1]
    if layers == 2:
        flow = True
    l_seq = sequence.shape[2]

    sequence_h_flipped = np.zeros_like(sequence)
    for t in range(l_seq):
        for l in range(layers):
            original = sequence[0, l, t, :, :]
            flipped = np.fliplr(original.copy())
            if flow and l == 0:
                flipped *= -1
                flipped += 1
            sequence_h_flipped[0, l, t, :, :] = flipped
    return sequence_h_flipped


def visualize_sequence(flow, depth, rgb, masks, label=0., step=2, run=None):
    """
    Visualizes the available data for a given sequence. Note, only one sequence can be put into this functions.

    Args:
        flow: the flow data. Shape (vx, vy) x (T/step) x (img height) x (img width)
        depth: the depth data. Shape (depth) x (T/step) x (img height) x (img width)
        rgb: the rgb data. Shape (red, green, blue) x (T/step) x (img height) x (img width)
        label: the ground truth label assigned to this sequence
        step: the number of frames that should be skipped in between frames that are kept
        run: the number of the run
    """
    imgs = []
    skip_step = 1
    # if step == 1:
    #     skip_step = 2
    # elif step == 2:
    #     skip_step = 1
    # else:
    #     raise ValueError("Visualization not yet implemented for step values higher than 2")

    if flow is not None:
        for k in range(8):
            imgs.append(flow[0, k * skip_step, :, :])
        for k in range(8):
            imgs.append(flow[1, k * skip_step, :, :])
    if depth is not None:
        for k in range(8):
            imgs.append(depth[0, k * skip_step, :, :])
    if rgb is not None:
        for k in range(8):
            imgs.append(np.transpose(
                rgb[:, k * skip_step, :, :], (1, 2, 0)).astype(int))
    if masks is not None:
        for k in range(8):
            imgs.append(masks[0, k * skip_step, :, :])

    fig = plt.figure(figsize=(12., 6.))
    grid = ImageGrid(fig, 111, nrows_ncols=(5, 8), axes_pad=0.1)
    if label == 0:
        annotate = 'Safe'
    elif label == 1:
        annotate = 'Act'
    elif label == 2:
        annotate = 'Urgent'

    fig.suptitle(f"Classification: {annotate}")
    # if run is None:
    #     fig.suptitle("Risk for this sequence: {:.2f}".format(label))
    # else:
    #     fig.suptitle("Risk for last sequence of run {}: {:.2f}".format(run, label))

    for ax, im in zip(grid, imgs):
        # Iterating over the grid returns the Axes.
        ax.imshow(im)

    plt.show()


def visualize_risk(flow, depth, rgb, masks, labels, step, threshold=0.5):
    """
    Prepares the data for the format required by the plotting function.
    Possibility to plot only specific risks. For instance, if threshold is 1, it only plots the sequences where the last
    frame is a collision frame.

    Args:
        flow: the flow data. Shape (# sequences) x (vx, vy) x (T/step) x (img height) x (img width)
        depth: the depth data. Shape (# sequences) x (depth) x (T/step) x (img height) x (img width)
        rgb: the rgb data. Shape (# sequences) x (red, green, blue) x (T/step) x (img height) x (img width)
        labels: the ground truth labels. Shape (# sequences) x (labels)
        step: the number of frames that should be skipped in between frames that are kept
        threshold:
    """
    num_sequences = labels.shape[0]
    # flow = flow.numpy()
    # depth = depth.numpy()
    # rgb = rgb.numpy()

    for i in range(num_sequences):
        if i % 1 == 0:
            if len(flow.shape) < 4:
                run_flow = None
            else:
                run_flow = flow[i, :, :, :, :]
            if len(depth.shape) < 4:
                run_depth = None
            else:
                run_depth = depth[i, :, :, :, :]
            if len(rgb.shape) < 4:
                run_rgb = None
            else:
                run_rgb = rgb[i, :, :, :, :]
            if len(masks.shape) < 4:
                run_masks = None
            else:
                run_masks = masks[i, :, :, :, :]
            label = labels[i]
            if label == 0 or label == 1:
                visualize_sequence(run_flow, run_depth,
                                   run_rgb, run_masks, label, step)