import csv
import numpy as np
import time
import torch
import matplotlib.pyplot as plt
import torchvision
from sklearn.metrics import r2_score, mean_squared_error, balanced_accuracy_score, accuracy_score, f1_score, precision_recall_fscore_support, roc_curve, auc

from torch.autograd import Variable
from torch.utils.data import DataLoader
import os
import json
from pathlib import Path
from sys import platform
import matplotlib


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class Logger(object):

    def __init__(self, path, header):
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


def perform_trial_run(train_data_loader, model, writer=None):
    print("*" * 79)
    print("Performing one trial run, showing sizes throughout the forward pass")

    model = model.train()

    data_iter = iter(train_data_loader)
    trial_batch = data_iter.next()
    trial_flow, trial_depth, trial_rgb, trial_masks, trial_labels = trial_batch

    if torch.cuda.is_available():
        trial_flow = trial_flow.cuda()
        trial_depth = trial_depth.cuda()
        trial_rgb = trial_rgb.cuda()
        trial_masks = trial_masks.cuda()

    # first_sequence_imgs = trial_feats[:, 2, 0, :, :]  # depth only
    # img_grid = torchvision.utils.make_grid(first_sequence_imgs)
    # writer.add_image('random_inputs', img_grid)
    # writer.add_graph(model, trial_flow, trial_depth, trial_rgb)

    t_start = time.time()
    output = model(trial_flow, trial_depth, trial_rgb, trial_masks)
    print("Computation time: {:.5f} s".format(time.time() - t_start))
    print("*" * 79)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def plot_metrics(x, y, xlabel="", ylabel='', title=''):
    plt.figure()
    plt.plot(x, y, lw=2, c='red')
    plt.grid()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def calculate_accuracy(outputs, targets, threshold=0.25):
    outputs = outputs.cpu().detach().numpy()
    predictions = np.zeros_like(outputs)
    negative_indx = outputs < threshold
    positive_indx = outputs >= threshold
    predictions[negative_indx] = 0
    predictions[positive_indx] = 1

    targets = targets.cpu().detach().numpy()
    labels = np.zeros_like(targets)
    negative_indx = targets < threshold
    positive_indx = targets >= threshold
    labels[negative_indx] = 0
    labels[positive_indx] = 1

    num_correct = np.count_nonzero(predictions == labels)
    num_total = targets.size
    acc = num_correct / num_total * 100
    return acc


def get_performance(prediction, ground_truth, rmse_range=False, range1=0.1, range2=0.5):
    prediction = np.array(prediction)
    ground_truth = np.array(ground_truth)
    mse = mean_squared_error(y_true=ground_truth, y_pred=prediction)
    rmse = mean_squared_error(y_true=ground_truth, y_pred=prediction, squared=False)
    r2 = r2_score(y_true=ground_truth, y_pred=prediction)
    if rmse_range:
        rmse_ranges = [mean_squared_error(y_true=ground_truth[ground_truth < range1],
                                          y_pred=prediction[ground_truth < range1], squared=False),
                       mean_squared_error(y_true=ground_truth[(ground_truth >= range1) & (ground_truth < range2)],
                                          y_pred=prediction[(ground_truth >= range1) & (ground_truth < range2)],
                                          squared=False),
                       mean_squared_error(y_true=ground_truth[ground_truth >= range2],
                                          y_pred=prediction[ground_truth >= range2], squared=False)]
        return mse, rmse, r2, rmse_ranges
    else:
        return mse, rmse, r2


def get_classification_metrics(prediction, ground_truth):
    if type(ground_truth) == type(list):
        ground_truth = np.array(ground_truth)
    if type(prediction) == type(list):
        prediction = np.array(prediction)

    idx = np.where(ground_truth == -1)
    ground_truth = np.delete(ground_truth, idx)
    prediction = np.delete(prediction, idx)
    balanced_acc = balanced_accuracy_score(ground_truth, prediction)
    precision, recall, fscore, _ = precision_recall_fscore_support(ground_truth, prediction, average='weighted', zero_division=0)
    return balanced_acc, precision, recall, fscore


def debug_print(targets, predictions, probabilities=None, mode=None):
    print('-'*79 + '\n' + f"{mode}")
    print(f"{targets}: \t All targets")
    print(f"{predictions}: \t All predictions")
    # probabilities = probabilities.cpu().detach().numpy()
    # print(f"{probabilities[0, :]}: \t Probabilities for 1st sequence (target={targets[0]})")
    # print(f"{probabilities[1, :]}: \t Probabilities for 2nd sequence (target={targets[1]})")
    # print(f"{probabilities[-2, :]}: \t Probabilities for 2nd to last sequence (target={targets[-2]})")
    # print(f"{probabilities[-1, :]}: \t Probabilities for last sequence (target={targets[-1]})")
    print('-' * 79)


def print_mem_usage():
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r - a  # free inside reserved
    print(f"GPU memory. Reserved: {r/1e8:.1f}. Allocated: {a/1e8:.1f}. Free: {f/1e8:.1f}")


def keep_strings_containing(original_list, string_to_keep):
    new_list = []
    for string in original_list:
        if string_to_keep in string:
            new_list.append(string)

    return new_list


def get_roc_auc_performance(labels, collision_probs):
    labels = np.array(labels)
    col_p = np.array(collision_probs)

    idxs = np.where(labels == -1)
    col_p = np.delete(col_p, idxs)
    labels = np.delete(labels, idxs)
    labels[np.where(labels == 2)] = 1

    fpr, tpr, thresholds = roc_curve(y_true=labels, y_score=col_p)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc, thresholds