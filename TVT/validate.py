import numpy as np
import torch
from torch.autograd import Variable
from TVT.utils import *
import time


def val_epoch(epoch, data_loader, model, criterion, writer, logger):
    model.eval()

    losses = AverageMeter()
    accs = AverageMeter()
    precs = AverageMeter()
    recalls = AverageMeter()
    fscores = AverageMeter()
    all_preds = []
    all_targets = []

    for i, (input_flow, input_depth, input_rgb, input_masks, targets) in enumerate(data_loader):
        t1 = time.time()

        if torch.cuda.is_available():
            targets = targets.cuda()
            input_flow = input_flow.cuda()
            input_depth = input_depth.cuda()
            input_rgb = input_rgb.cuda()
            input_masks = input_masks.cuda()

        with torch.no_grad():
            outputs = model(input_flow, input_depth, input_rgb, input_masks)

        loss = criterion(outputs, targets.type(torch.int64))
        writer.add_scalar('validation loss', loss.data, epoch * len(data_loader) + i)

        values, preds = torch.max(outputs, 1)
        targets_list = targets.cpu().detach().tolist()
        preds_list = preds.cpu().detach().tolist()
        all_preds.extend(preds_list)
        all_targets.extend(targets_list)

        losses.update(loss.data, len(targets_list))
        bal_acc, precision, recall, fscore = get_classification_metrics(preds_list, targets_list)
        accs.update(bal_acc, len(targets_list))
        precs.update(precision, len(targets_list))
        recalls.update(recall, len(targets_list))
        fscores.update(fscore, len(targets_list))

        # if torch.cuda.is_available():
        #     if i % 25 == 0:
        #         print(f"Epoch: [{epoch}][{i}/{len(data_loader)}]\t Time: "
        #               f"{time.time() - t1:.2f} s\t Val_Loss: {loss:.3f} \t Val_Accuracy (balanced) {bal_acc:.3f}\t"
        #               f"Val_Precision {precision:.3f}\t Val_Recall {recall:.3f} \t Val_F-score {fscore:.3f}")
        # else:
        #     print(f"Epoch: [{epoch}][{i}/{len(data_loader)}]\t Time: "
        #           f"{time.time() - t1:.2f} s\t Val_Loss: {loss:.3f} \t Val_Accuracy (balanced) {bal_acc:.3f}\t"
        #           f"Val_Precision {precision:.3f}\t Val_Recall {recall:.3f} \t Val_F-score {fscore:.3f}")

    print(f"Epoch: [{epoch}][mean]\t Val_Loss: {losses.avg.item():.3f} \t Val_Accuracy (balanced) {accs.avg.item():.3f}\t"
          f"Val_Precision {precs.avg.item():.3f}\t Val_Recall {recalls.avg.item():.3f} \t Val_F-score {fscores.avg.item():.3f}"
          f"\t Average predicted label: {np.mean(np.array(all_preds)):.3f}"
          f"\t Average actual label: {np.mean(np.array(all_targets)):.3f}")

    logger.log({
        'epoch': epoch,
        'loss': losses.avg.item(),
        'acc': accs.avg.item(),
        'prec': precs.avg.item(),
        'recall': recalls.avg.item(),
        'fscore': fscores.avg.item()
    })

    return fscores.avg.item(), losses.avg.item()