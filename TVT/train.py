import numpy as np
import torch
from TVT.utils import *
import time
from torch.autograd import Variable
import matplotlib.pyplot as plt


def train_epoch(epoch, data_loader, model, criterion, optimizer, scheduler, epoch_logger, batch_logger):
    print("\n" * 1, "-"*79, "\n" * 1)
    print("Epoch {}".format(epoch))
    print("\n" * 1)
    model.train()

    losses = AverageMeter()
    accs = AverageMeter()
    precs = AverageMeter()
    recalls = AverageMeter()
    fscores = AverageMeter()
    all_preds = []
    all_targets = []

    for i, (input_flow, input_depth, input_rgb, input_mask, targets) in enumerate(data_loader):
        t1 = time.time()

        if torch.cuda.is_available():
            targets = targets.cuda()
            input_flow = input_flow.cuda()
            input_depth = input_depth.cuda()
            input_rgb = input_rgb.cuda()
            input_mask = input_mask.cuda()

        input_flow = Variable(input_flow)
        input_depth = Variable(input_depth)
        input_rgb = Variable(input_rgb)
        input_mask = Variable(input_mask)
        targets = Variable(targets).type(torch.int64)
        outputs = model(input_flow, input_depth, input_rgb, input_mask)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        values, preds = torch.max(outputs, 1) # preds is the index of the max value in the output tensor, since this is a binary classification problem, preds will be 0 or 1
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
        #         print(f"Epoch: [{epoch}][{i}/{len(data_loader)}]\t lr: {optimizer.param_groups[0]['lr']:.8f}\t Time: "
        #               f"{time.time() - t1:.2f} s\t Loss: {loss:.3f} \t Accuracy (balanced) {bal_acc:.3f}\t"
        #               f"Precision {precision:.3f}\t Recall {recall:.3f} \t F-score {fscore:.3f}")
        # else:
        #     print(f"Epoch: [{epoch}][{i}/{len(data_loader)}]\t lr: {optimizer.param_groups[0]['lr']:.5f}\t Time: "
        #           f"{time.time() - t1:.2f} s\t Loss: {loss:.3f} \t Accuracy (balanced) {bal_acc:.3f}\t"
        #           f"Precision {precision:.3f}\t Recall {recall:.3f} \t F-score {fscore:.3f}")

        batch_logger.log({
            'epoch': epoch,
            'batch': i + 1,
            'iter': (epoch - 1) * len(data_loader) + (i + 1) + 1,
            'loss': losses.val.item(),
            'lr': optimizer.param_groups[0]['lr'],
        })

    # writer.add_scalar('training loss', losses.avg.item(), epoch)
    # writer.add_scalar('acc', accs.avg.item(), epoch)
    # writer.add_scalar('f1', fscores.avg.item(), epoch)

    print(f"Epoch: [{epoch}][mean]\t Loss: {losses.avg.item():.3f} \t Accuracy (balanced) {accs.avg.item():.3f}\t"
          f"Precision {precs.avg.item():.3f}\t Recall {recalls.avg.item():.3f} \t F-score {fscores.avg.item():.3f}"
          f"\t Average predicted label: {np.mean(np.array(all_preds)):.3f}"
          f"\t Average actual label: {np.mean(np.array(all_targets)):.3f}")

    scheduler.step()
    epoch_logger.log({
        'epoch': epoch,
        'loss': losses.avg.item(),
        'lr': optimizer.param_groups[0]['lr'],
        'acc': accs.avg.item(),
        'prec': precs.avg.item(),
        'recall': recalls.avg.item(),
        'fscore': fscores.avg.item()
    })

    return fscores.avg.item(), losses.avg.item()