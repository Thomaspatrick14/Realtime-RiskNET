import torch
import time
from tRTfiles.yolo_tensorrt_engine import do_inference
import numpy as np

#############################################################
####################### Not optimized #######################
#############################################################

def test(masks, model, logger=None, print_pred=False, return_probs=False):
    t1 = time.time()
    predictions = []
    col_probs = []
    if torch.cuda.is_available():
        masks = masks.cuda()

    with torch.no_grad():
        outputs = model(masks)
    preds = torch.max(outputs, 1).indices # preds is the index of the max value in the output tensor, since this is a binary classification problem, preds will be 0 or 1
    probs = outputs[:, 1].cpu().detach().tolist()

    col_probs = probs
    predictions = preds.cpu().detach().tolist()

    if print_pred:
        print(f"Prediction: {predictions}")

    t_total = time.time() - t1

    print(f"Inference time {t_total:.4} s")
    if return_probs:
        return predictions, col_probs, t_total
    else:
        return predictions, t_total

#############################################################
###################    TensorRT   ###########################
#############################################################

# def test(masks, context, tensorrt, logger=None, print_pred=False, return_probs=False):
#     t1 = time.time()
#     inputs, outputs, bindings, stream = tensorrt
#     predictions = []
#     col_probs = []
    
#     masks = masks.cpu()
#     masks_np = masks.numpy().astype(np.float32).ravel()
#     np.copyto(inputs[0]['host'], masks_np)

#     # Perform inference
#     outputs_host = do_inference(context, bindings, inputs, outputs, stream)

#     # Convert output to torch tensor
#     outputs = torch.tensor(outputs_host[0].reshape(-1, 2))

#     preds = torch.max(outputs, 1).indices # preds is the index of the max value in the output tensor, since this is a binary classification problem, preds will be 0 or 1
#     probs = outputs[:, 1].cpu().detach().tolist()

#     col_probs = probs
#     predictions = preds.cpu().detach().tolist()

#     if print_pred:
#         print(f"Prediction: {predictions}")

#     t_total = time.time() - t1

#     print(f"Inference time {t_total:.4} s")
#     if return_probs:
#         return predictions, col_probs, t_total
#     else:
#         return predictions, t_total