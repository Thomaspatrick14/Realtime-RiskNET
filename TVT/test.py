import numpy as np
import torch
import time
from torch.autograd import Variable


def test(masks, model, logger=None, print_pred=False, return_probs=False):
    t1 = time.time()
    predictions = []
    col_probs = []
    if torch.cuda.is_available():
        masks = masks.cuda()

    with torch.no_grad():
        outputs = model(masks)

    preds = torch.max(outputs, 1).indices
    probs = outputs[:, 1].cpu().detach().tolist()

    col_probs = probs
    predictions = preds.tolist()

    if print_pred:
        print(f"Prediction: {predictions}")

    t_total = time.time() - t1

    print(f"Inference time {t_total:.4} s")
    if return_probs:
        return predictions, col_probs, t_total
    else:
        return predictions, t_total