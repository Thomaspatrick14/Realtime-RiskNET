import torch
# from yolodet import det_model
from torch.autograd import Variable
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import torchvision
import time
import argparse
import numpy as np
import math
import os
import json
from pathlib import Path
from sys import platform
from datetime import datetime
from torchvision import transforms
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from dataset.Realtimetest.real_inference_dataset import get_masks
from dataset.Realtimetest.Detector import detect
from TVT.train import train_epoch
from TVT.validate import val_epoch
from TVT.test import test
from pred_models.create_model import get_model
from dataset.dataset import SiemensDataset, MaskNoise
from TVT.utils import *
import cv2


if platform == "win32":
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # unsupported workaround for OMP: Error #15 on WIN32


# debugging strange error
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.allow_tf32 = True



# if __name__ == '__main__':
parser = argparse.ArgumentParser(description='Passing arguments to main risk estimation file.')

# Run parameters
parser.add_argument("--run_name", type=str, default="experiment_1",
                    help='Name of the run to be tested or evaluated')
parser.add_argument("--input", default=['flow'], nargs='+',
                    help='List of inputs that the model should take into account. Default: [flow], options: '
                         '[flow, depth, rgb, mask]')
parser.add_argument("--n_backbones", type=int, default=1,
                    help='Number of backbone to be used, max number of backbones = number of inputs.')
parser.add_argument("--backbone", default="ResNext18",
                    help='Backbone to be used. Options: ResNext18, ResNext50, ResNext101 ')
parser.add_argument("--train", default=False, action='store_true',
                    help='Indicates whether the model should be trained')
parser.add_argument("--visualize", default=False, action='store_true',
                    help='Visualizes some sequences from the dataset.')
parser.add_argument("--test_exp", default=False, action='store_true',
                    help='Indicates whether the model should be tested on all saved training checkpoints')

# Attention mechanism
parser.add_argument("--conv1_out", type=int, default=8,
                    help='Number of feature maps in the soft attention mechanism')
parser.add_argument("--batch_norm", default=False, action='store_true',
                    help='Indicates whether to use batch normalization in the soft attention mechanism.')
parser.add_argument("--mask_method", default="binary",
                    help='Type of masks to be used. Options for training: binary, gaussian. Options for real world '
                         'dataset: case1, case2, case3, case4')
parser.add_argument("--mask_prior", default="none",
                    help='Type of prior to be used on the masks. Options: none, road, car')

# Training parameters
parser.add_argument("--n_epochs", type=int, default=2,
                    help='Number of epochs to train the model')
parser.add_argument("--p_noise", type=int, default=20,
                    help='Probability of noise being added to the model.')
parser.add_argument("--lr", type=float, default=0.001,
                    help='Learning rate')
parser.add_argument("--l2_norm", type=float, default=0.00001,
                    help='L2 decay norm')
parser.add_argument("--gamma", type=float, default=0.5,
                    help='Learning rate exponential decay factor. Gamma = 1 --> no decaying LR')
parser.add_argument("--lr_step", type=int, default=20,
                    help='Learning rate exponential decay factor. Gamma = 1 --> no decaying LR')
parser.add_argument("--balance_weights", default=True, action='store_false',
                    help='Indicates whether optimizer should compensate for class imbalances')

# Data parameters
# TODO: update code to actually use the dataset string
parser.add_argument("--dataset", default="raw",
                    help='The folder in which the dataset is located. Options: raw, processed1, processed3, processed4,'
                         'real_world')
parser.add_argument("--batch_size", type=int, default=8,
                    help='Batch size of the training dataloader')
parser.add_argument("--downscale_factor", type=int, default=8,
                    help='Factor indicating how much the original images should be downscaled.')
parser.add_argument("--T", type=int, default=16,
                    help='History time (in frames). So at 20 FPS, T=16 gives 0.8s history time. Should be 8, 16, 32')
parser.add_argument("--sample_strat", default="tim",
                    help='The strategy that is to be used for sampling. Choose from tim, jp, fair')
parser.add_argument("--threshold", type=int, default=0.6,
                    help='the threshold for the fraction of the sequence which is to be unsafe')
parser.add_argument("--skip", type=float, default=0.5,
                    help='factor which is to be multiplied to T which decides the amount of frames to be skipped')
parser.add_argument("--step", type=int, default=2,
                    help='Keep 1 frame out of step frames. So if step = 1, no frames are dropped. if step = 2, 50% of '
                         'frames are dropped, etc. Should be 1, 2, 4, etc.')
parser.add_argument("--k_fold", type=int, default=0,
                    help='Indicates which dataset fold to use. Note: if -1, specify the train/val/test runs.')
parser.add_argument("--h_flip", default=False, action='store_true',
                    help='Indicates whether the model should do data augmentation by doing horizontal flipping')
parser.add_argument("--tiny_dataset", default=False, action='store_true',
                    help='If True, it will use only a small part of dataset. Can be used for debugging.')
parser.add_argument("--single_run", default=-1, type=int,
                    help='If run is given (must be int), only data of this run will be gathered.')


args = parser.parse_args()
print(f"PyTorch Version {torch.__version__}")
print(f"Start time: {datetime.now()}")
print("\n" * 5, "-"*79, "\n", "-"*79)
print("Args for this run: \n", args)

# Directory checking
run_path = './runs/' + args.run_name
Path(run_path).mkdir(parents=True, exist_ok=True)



if args.train:
    with open(run_path + '/args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
        print(f"Saved run parameters to {run_path + '/args.txt'}")


Path(run_path + '/tensorboard').mkdir(parents=True, exist_ok=True)
tensorboard_files = os.listdir(run_path + '/tensorboard')
if (len(tensorboard_files) > 0) and args.train:
    print("Removing {} old tensorboard file(s)".format(len(tensorboard_files)))
    for file_name in tensorboard_files:
        file_path = run_path + '/tensorboard/' + file_name
        os.remove(file_path)


########################################################################################################################################
                                                ######### Training & Validation loop ###########
########################################################################################################################################

if args.train:
    # tensorboard
    writer = SummaryWriter(run_path + '/tensorboard')

    # Load the model
    model = get_model(args)

    print('-'*79 + "\nLoading training set")
    train_set = SiemensDataset(mode='train', args=args, transform=transforms.Compose([MaskNoise(p_noise=args.p_noise)]))
    train_data_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True)
    # print_mem_usage()
    
    print('-'*79 + "\nLoading validation set")
    val_set = SiemensDataset(mode='val', args=args)
    val_data_loader = DataLoader(dataset=val_set, batch_size=args.batch_size, shuffle=True)

    # assigning weights to each sample to ensure even classes (Only for training)
    if args.balance_weights:
        # this section is for 3 class classification:
        # if len(train_set.counts) != 3:
        #     weights = torch.Tensor([1., 1., 1.])  # don't set any weights, because some labels are missing.
        # highest_count = np.max(train_set.counts)
        # weight_safe = highest_count / train_set.counts[0]
        # weight_act = highest_count / train_set.counts[1]
        # weight_urgent = highest_count / train_set.counts[2]
        # weights = torch.Tensor([weight_safe, weight_act, weight_urgent])

        # this section is only for binary classification:
        highest_count = np.max(train_set.counts)
        weight_safe = highest_count / train_set.counts[0]
        weight_urgent = highest_count / train_set.counts[1]
        weights = torch.Tensor([weight_safe, weight_urgent])
    else:
        weights = torch.Tensor([1., 1.])

    print(f"Using the following weights in the loss function: {weights}")
    criterion = nn.CrossEntropyLoss(weights)
    if torch.cuda.is_available():
        print("CUDA available: loading the loss function on the GPU")
        criterion.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_norm)
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.gamma)

    train_logger = Logger('./runs/' + args.run_name + '/train.log', ['epoch', 'loss', 'lr', 'acc', 'prec', 'recall',
                                                                     'fscore'])
    train_batch_logger = Logger('./runs/' + args.run_name +  '/train_batch.log', ['epoch', 'batch', 'iter', 'loss',
                                                                                  'lr'])
    val_logger = Logger('./runs/' + args.run_name +  '/val.log', ['epoch', 'loss', 'acc', 'prec', 'recall', 'fscore'])

    best_fscore = 0
    best_loss = 999999
    best_epoch = 0
    t_start_train = time.time()
    train_losses = []
    val_losses = []
    train_fscores = []
    val_fscores = []
    for epoch in range(args.n_epochs):
        t_start_epoch = time.time()
        is_best = False
        writer.add_scalar('Learning rate', optimizer.param_groups[0]['lr'], epoch)
        fscore, loss = train_epoch(epoch, train_data_loader, model, criterion, optimizer, scheduler, writer,
                                   train_logger, train_batch_logger)
        save_name = '/checkpoint_' + str(epoch) + '.pth'
        torch.save(model.state_dict(), run_path + save_name)
        print(f"Saved model {save_name}")
        # is_best = loss < best_loss
        print("Epoch {} trained in {:.2f} s".format(epoch, time.time() - t_start_epoch))
        print("\n" * 1)
        val_fscore, val_loss = val_epoch(epoch, val_data_loader, model, criterion, writer, val_logger)
        print("Epoch {} validated in {:.2f} s".format(epoch, time.time() - t_start_epoch))

        train_losses.append(loss)
        val_losses.append(val_loss)
        train_fscores.append(fscore)
        val_fscores.append(val_fscore)

        if (val_fscore > best_fscore):
            is_best = True
            torch.save(model.state_dict(), run_path + '/checkpoint_' + 'best_epoch.pth')

        if is_best:
            print(f"New best model with Val F-score {val_fscore:.3f}. Previous best Val F-score {best_fscore:.3f}"
                  f" (epoch {best_epoch})")
            best_fscore = val_fscore
            # print(f"New best model with loss {val_loss:.3f}. Previous best loss {best_loss:.3f}"
            #       f" (epoch {best_epoch})")
            # best_loss = val_loss
            best_epoch = epoch
        else:
            print(f"Model not better than previous best of F-score {best_fscore:.3f} (epoch {best_epoch})")
            # print(f"Loss {val_loss:.3f} not better than previous best loss {best_loss:.3f} (epoch {best_epoch})")
    
    epochs = list(range(args.n_epochs))  
    # Plot the training and validation losses
    plt.figure()
    plt.plot(train_losses, label='Training Loss', marker='o', linestyle='-', color='b')
    plt.plot(val_losses, label='Validation Loss', marker='o', linestyle='-', color='r')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xticks(epochs)
    plt.legend()
    plt.title('Training and Validation Loss Over Epochs')
    plt.grid(True)
    plt.savefig('/storage/users/j.p.udhayakumar/results/' + args.run_name + 'loss_plot.png')
    plt.close()

    # Plot the training and validation F-score
    plt.figure()
    plt.plot(train_fscores, label='Training F-score', marker='o', linestyle='-', color='b')
    plt.plot(val_fscores, label='Validation F-score', marker='o', linestyle='-', color='r')
    plt.xlabel('Epoch')
    plt.ylabel('F-score')
    plt.xticks(epochs)
    plt.legend()
    plt.title('Training and Validation F-score Over Epochs')
    plt.grid(True)
    plt.savefig('/storage/users/j.p.udhayakumar/results/' + args.run_name + 'fscore_plot.png')
    plt.close()

    print("Total train time: {:.1f} minutes".format((time.time() - t_start_train) / 60))

    writer.close()

########################################################################################################################################
                                                ######### Testing loop ###########
########################################################################################################################################


if (not args.train) or args.test_exp:

    with open(run_path + '/args.txt') as file:
        train_args = json.load(file)
        
    assert args.input == train_args['input'], \
        f"Evaluation input to model is {args.input} but the model was trained with{train_args['input']}"
    assert args.n_backbones == train_args['n_backbones'], \
        f"Evaluation n_backbones of model is {args.n_backbones} but the model was trained with {train_args['n_backbones']}" \
        f"backbones"
    assert args.backbone in [train_args['backbone']], \
        f"Evaluation backbone of model is {args.backbone} but the model was trained with {train_args['backbone']}"
    assert args.downscale_factor == train_args['downscale_factor'], \
        f"Evaluation dataset downscale factor is {args.downscale_factor} but the model was trained with" \
        f" dataset downscale factor {train_args['downscale_factor']}"
    assert args.T == train_args['T'], \
        f"Evaluation sequence time T is {args.T} but the model was trained with sequence time T {train_args['T']}"
    assert args.step == train_args['step'], \
        f"Evaluation step is {args.step} but the model was trained with step {train_args['step']}"
    assert args.k_fold == train_args['k_fold'], \
        f"Evaluation fold is {args.k_fold} but the model was trained on fold {train_args['k_fold']}"

    print("\n" * 5, "-"*79, "\n", "-"*79)

    img_size = [int(960 / args.downscale_factor),
                    int(1280 / args.downscale_factor)]  # [120, 160] for downscale_factor = 8

    # Load the model
    pred_model = get_model(args)

    if platform == "win32":
        pred_model.load_state_dict(torch.load(run_path + '/checkpoint_best_epoch.pth', map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load(run_path + '/checkpoint_best_epoch.pth'))
    pred_model.eval()
    Path(run_path).mkdir(parents=True, exist_ok=True)

    # Load the Detection model
    print(f"Loading the Detection model")
    det_model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
    if torch.cuda.is_available():
        print("CUDA available: loading the detection model on the GPU")
        det_model = det_model.cuda()
    det_model.eval()

    folder_path = "C:/JP/TUe/2nd year Internship and thesis/Thesis/Code/Thesis"

    # Cold start the model
    print('-'*79 + "\nCold starting the models...")
    cold_frame = [np.ones((360, 480, 3)) for _ in range(8)]
    cold_frame = detect(cold_frame, folder_path, det_model)
    cold_masks, _, _ = get_masks(cold_frame, img_size, args.mask_method, args.mask_prior, viz=True)
    cold_masks = torch.tensor(cold_masks).unsqueeze(0).float()
    test(cold_masks, pred_model)

    # Start the camera/video
    print('-'*79 + "\nStarting Camera...")
    
    video_path = os.path.join(folder_path, "output_c2bt.mp4")  # Replace with the path to your video file
    cap = cv2.VideoCapture(video_path) # 0 for webcam

    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video FPS: {fps}")
    # if fps % 10 != 0:
    #     print("\nFPS is not a multiple of 10. Please use a source with FPS as a multiple of 10")
    #     exit()

    multiple = int(fps / 10) # 10 is the frame rate of the model

    video_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps

    frame_count = 0
    first = True
    counter = 0
    
    
    # Get dimensions of input frames
    height, width = 360, 480

    # Define video codec and create VideoWriter object
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # video_writer = cv2.VideoWriter('C:/Users/up650/Downloads/output_video.mp4', fourcc, 10, (width*2, height*2))
    probs_list = []
    t1 = time.time()
    while True:
        t2 = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count == 0 or frame_count % multiple == 0:
            # reshaping the frame to the required size
            #if frame is not 480x360, resize it
            if frame.shape[0] != 360 or frame.shape[1] != 480:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
            mask, processed_boxes, dboxes = get_masks(detect(frame, folder_path, det_model), img_size, args.mask_method, args.mask_prior, viz=True)
            dmask = mask[0,0,:,:]
            dmask = cv2.resize(dmask, (width, height))
            dmask = dmask.astype(np.uint8) * 255  # Multiply by 255 to convert bool to uint8
            dmask = np.expand_dims(dmask, axis=-1)
            dmask = np.repeat(dmask, 3, axis=-1)  # Repeat the dimension three times
            dmask = cv2.putText(dmask, f"Attention Masks", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            # Display dboxes overlayed on current frame
            frame_with_dboxes = frame.copy()
            for box1 in dboxes:
                x, y, x2, y2 = box1
                cv2.rectangle(frame_with_dboxes, (x, y), (x2, y2), (0, 255, 0), 2)
            frame_with_dboxes = cv2.putText(frame_with_dboxes, f"Actual Detections", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Display processed_boxes overlayed on current frame
            frame_with_processed_boxes = frame.copy()
            for box2 in processed_boxes:
                x, y, x2, y2 = box2
                cv2.rectangle(frame_with_processed_boxes, (x, y), (x2, y2), (255, 0, 0), 2)
            frame_with_processed_boxes = cv2.putText(frame_with_processed_boxes, f"Filtered Detections", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            mask = torch.tensor(mask).unsqueeze(0).float()

            if first:
                masks = mask
                first = False
            else:
                masks = torch.cat((masks, mask), dim=2)

            if masks.shape[2] == 8:
                predictions, probs = test(masks, pred_model, return_probs=True)
                probs_list.append(probs[0])
                print(f"Prediction: {predictions}")
                masks = masks[:,:,1:,:,:]
                print(f"Time for seq: {time.time() - t2:.4} s")    
                counter += 1
                print(f"Sequence no.: {counter}")
                if counter == 1:
                    print(f"Time for the first seq (8 detections + 1 prediction): {time.time() - t1:.4} s")
                    
                # Visualize the prediction for each frame
                frame = cv2.putText(frame, f"Prediction: {predictions[0]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                combined_frame = np.hstack((np.vstack((frame, frame_with_processed_boxes)),np.vstack((frame_with_dboxes, dmask))))

                # Create the plot
                plt.plot(probs_list, color='blue')
                plt.xlabel('Time step')
                plt.ylabel('Probability of Collision Risk')
                plt.title('Risk over time')
                plt.ylim(0, 1)
                plot = plt.gcf()
                plot.canvas.draw()
                plot_array = np.array(plot.canvas.renderer.buffer_rgba())

                cv2.imshow('Plot', plot_array)
                cv2.imshow('Visualizer', combined_frame)

                # Write the combined frame to the video file
                # video_writer.write(combined_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        frame_count += 1

    print(f"\nVideo Duration: {video_duration} s \nTotal processing time: {time.time() - t1:.4} s")
    cap.release()
    cv2.destroyAllWindows()
    # video_writer.release()

print("-"*79, "\n", "-"*79, "\n" * 5)

# python main.py --run_name Thesis_test --input mask --backbone ResNext18 --mask_method "case4"