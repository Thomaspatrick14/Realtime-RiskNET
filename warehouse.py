import cv2
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from dataset.Realtimetest.real_inference_dataset import get_masks
from dataset.Realtimetest.Detector import detect
from TVT.test import test
import os
from TVT.utils import *

##############################################################################################
#################################       Append frames       ##################################
##############################################################################################
def append_frames(folder_path, det_model, pred_model, args, img_size, video_path):
    # Cold start the model
    print('-'*79 + "\nCold starting the models...")
    cold_frame = [np.ones((360, 480, 3)) for _ in range(8)]
    cold_frame = detect(cold_frame, folder_path, det_model)
    cold_masks = get_masks(cold_frame, img_size, args.mask_method, args.mask_prior)
    cold_masks = torch.tensor(cold_masks).unsqueeze(0).float()
    test(cold_masks, pred_model)

    # Start the camera/video
    print('-'*79 + "\nStarting Camera...")
    
    cap = cv2.VideoCapture(video_path) # 0 for webcam

    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video FPS: {fps}")
    # if fps % 10 != 0:
    #     print("\nFPS is not a multiple of 10. Please use a source with FPS as a multiple of 10")
    #     exit()

    multiple = int(fps / 10) # 10 is the frame rate of the model

    video_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps

    frame_count = 0
    frames_to_process = []
    first = True
    counter = 0
    total_neram = 0
    t1 = time.time()
    while True:
        t2 = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count == 0 or frame_count % multiple == 0:
            frames_to_process.append(frame)

            if len(frames_to_process) == 8:
                if first:
                    detections = detect(frames_to_process, folder_path, det_model)
                    masks = get_masks(detections, img_size, args.mask_method, args.mask_prior)
                    masks = torch.tensor(masks).unsqueeze(0).float()
                    predictions = test(masks, pred_model)
                    print(f"Prediction: {predictions}")
                    frames_to_process = frames_to_process[1:]
                    first = False
                    first_seq = time.time() - t1
                    print(f"Time for first seq (8 detections + 1 prediction): {first_seq:.4} s")
                else:
                    detections = detect(frames_to_process[-1], folder_path, det_model)
                    mask = get_masks(detections, img_size, args.mask_method, args.mask_prior)
                    mask = torch.tensor(mask).unsqueeze(0).float()
                    masks = masks[:,:,1:,:,:]
                    masks = torch.cat((masks, mask), dim=2)
                    predictions = test(masks, pred_model)
                    print(f"Prediction: {predictions}")
                    frames_to_process = frames_to_process[1:]
                
                counter += 1
                print(f"Sequence no.: {counter}\n")
                neram = time.time() - t2  # neram = time
                print(f"Time for seq: {neram:.4} s") 
                total_neram += neram
        
        if counter == 142: #for testing camera
            break
        
        frame_count += 1
            

    print(f"\nVideo Duration: {video_duration} s \nTotal processing time: {time.time() - t1:.4} s \nTime for first seq (8 detections + 1 prediction): {first_seq:.4} s \nAverage sequence time: {total_neram/counter:.4} s")
    cap.release()

##############################################################################################
#################################     Append detections     ##################################
##############################################################################################
def append_detections(folder_path, det_model, pred_model, args, img_size, video_path):
    # Cold start the model
    print('-'*79 + "\nCold starting the models...")
    cold_frame = [np.ones((360, 480, 3)) for _ in range(8)]
    cold_frame = detect(cold_frame, folder_path, det_model)
    cold_masks = get_masks(cold_frame, img_size, args.mask_method, args.mask_prior)
    cold_masks = torch.tensor(cold_masks).unsqueeze(0).float()
    test(cold_masks, pred_model)

    # Start the camera/video
    print('-'*79 + "\nStarting Camera...")
    
    cap = cv2.VideoCapture(video_path) # 0 for webcam

    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video FPS: {fps}")
    # if fps % 10 != 0:
    #     print("\nFPS is not a multiple of 10. Please use a source with FPS as a multiple of 10")
    #     exit()

    multiple = int(fps / 10) # 10 is the frame rate of the model

    video_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps

    frame_count = 0
    detections = []
    first = True
    counter = 0
    total_neram = 0
    t1 = time.time()
    while True:
        t2 = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count == 0 or frame_count % multiple == 0:
            if first:
                detections.append(detect(frame, folder_path, det_model)[0])
            else:
                dets = detect(frame, folder_path, det_model)

            if len(detections) == 8:
                if first:
                    masks = get_masks(detections, img_size, args.mask_method, args.mask_prior)
                    masks = torch.tensor(masks).unsqueeze(0).float()
                    predictions = test(masks, pred_model)
                    print(f"Prediction: {predictions}")
                    first = False
                    first_seq = time.time() - t1
                    print(f"Time for the first sequence (8 detections + 1 prediction): {first_seq:.4} s")
                else:
                    mask = get_masks(dets, img_size, args.mask_method, args.mask_prior)
                    mask = torch.tensor(mask).unsqueeze(0).float()
                    masks = masks[:,:,1:,:,:]
                    masks = torch.cat((masks, mask), dim=2)
                    predictions = test(masks, pred_model)
                    print(f"Prediction: {predictions}")
                
                counter += 1
                print(f"Sequence no.: {counter}")
                neram = time.time() - t2  # neram = time
                print(f"Time for seq: {neram:.4} s") 
                total_neram += neram
        
        # if counter == 142: #for testing camera
        #     break
        
        frame_count += 1

    print(f"\nVideo Duration: {video_duration} s \nTotal processing time: {time.time() - t1:.4} s \nTime for first seq (8 detections + 1 prediction): {first_seq:.4} s \nAverage sequence time: {total_neram / counter:.4} s")
    cap.release()

##############################################################################################
################################# Append detections + masks ##################################
##############################################################################################
def append_detections_masks(det_model, pred_model, args, img_size, video_path):
    # Cold start the models
    # print('-'*79 + "\nCold starting the models...")
    # cold_frame = np.ones((360, 480, 3))
    # cold_masks = get_masks(detect(cold_frame, det_model), img_size, args.mask_method, args.mask_prior)
    # cold_masks = torch.tensor(cold_masks).unsqueeze(0).float()
    # cold_masks = cold_masks.repeat(1, 1, 8, 1, 1)
    # test(cold_masks, pred_model)
    # test(cold_masks, pred_model)

    # Start the camera/video
    print('-'*79 + "\nStarting Camera...")
    
    cap = cv2.VideoCapture(video_path) # 0 for webcam

    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video FPS: {fps}")
    # if fps % 10 != 0:
    #     print("\nFPS is not a multiple of 10. Please use a source with FPS as a multiple of 10")
    #     exit()

    multiple = int(fps / 10) # 10 is the frame rate of the model

    if video_path == 0 or video_path == 1:
        video_duration = 0
    else:
        video_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps

    frame_count = 0
    total_neram = 0
    tpred = 0
    first = True
    counter = 0
    t1 = time.time()
    pred_list = []
    while True:
        t2 = time.time()
        ret, frame = cap.read()
        # print(f"time for reading frame: {time.time() - t2:.4} s")
        if not ret:
            break

        # if frame_count == 0 or frame_count % multiple == 0:
        mask = get_masks(detect(frame, det_model), img_size, args.mask_method, args.mask_prior)
        mask = torch.tensor(mask).unsqueeze(0).float()
        if first:
            masks = mask
            first = False
        else:
            masks = torch.cat((masks, mask), dim=2)

        if masks.shape[2] == 8:
            predictions, t_total = test(masks, pred_model)
            tpred += t_total
            pred_list.append(predictions[0])
            print(f"Prediction: {predictions}")
            masks = masks[:,:,1:,:,:]
            counter += 1
            neram = time.time() - t2  # neram = time
            total_neram += neram
            print(f"Sequence no.: {counter}")
        
            if counter == 1:
                first_seq = time.time() - t1
                print(f"Time for the first sequence (8 detections + 1 prediction): {first_seq:.4} s")
                print(f"Time for seq (1 det + 1 pred): {neram:.4} s")
            else:
                print(f"Time for seq (1 det + 1 pred): {neram:.4} s")
                
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #         break
        # if frame_count == 1500: #for testing camera
        #     break

        frame_count += 1
        
    # save_path = folder_path + '/csv' + '/ydid_clip5_predictions_labels_30to10_fps' + '.csv'
    # pred_labels = np.array([pred_list])
    # np.savetxt(str(save_path), pred_labels, delimiter=',')
    # print(f"Saved predictions and labels to {save_path}")

    print(f"\nVideo Duration: {video_duration} s \nTotal processing time: {time.time() - t1:.4} s"
          f"\nTime for first seq (8 detections + 1 prediction): {first_seq:.4} s" 
          f"\nAverage sequence time: {total_neram / counter:.4} s" 
          f"\nAverage prediction time: {tpred / counter:.4} s")
    cap.release()

##############################################################################################
############################ Append detections + masks (visualize) ###########################
##############################################################################################
def append_detections_masks_viz(det_model, pred_model, args, img_size, video_path):
    
    # Cold start the model
    viz=True
    print('-'*79 + "\nCold starting the models...")
    cold_frame = np.ones((360, 480, 3))
    cold_masks = get_masks(detect(cold_frame, det_model), img_size, args.mask_method, args.mask_prior)
    cold_masks = torch.tensor(cold_masks).unsqueeze(0).float()
    cold_masks = cold_masks.repeat(1, 1, 8, 1, 1)
    test(cold_masks, pred_model)
    test(cold_masks, pred_model)

    # Start the camera/video
    print('-'*79 + "\nStarting Camera...")
    
    cap = cv2.VideoCapture(video_path) # 0 for webcam

    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video FPS: {fps}")
    print(f"camera resolution: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)} x {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    # if fps % 10 != 0:
    #     print("\nFPS is not a multiple of 10. Please use a source with FPS as a multiple of 10")
    #     exit()

    multiple = int(fps / 10) # 10 is the frame rate of the model

    video_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps

    frame_count = 0
    total_neram = 0
    first = True
    counter = 0
    
    
    # Get dimensions of input frames
    height, width = 360, 480

    # Define video codec and create VideoWriter object
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # video_writer = cv2.VideoWriter('C:/Users/up650/Downloads/output_video.mp4', fourcc, 10, (width*2, height*2))

    # For ablation study
    # labels_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "labels.csv")
    # run_labels_data = np.loadtxt(labels_path, delimiter=',')
    # run_labels_data[run_labels_data == 1] = 0
    # run_labels_data[run_labels_data == 2] = 1
    
    probs_list = []
    preds_list = []
    # label_list = [] # for ablation study
    neram_list = []
    tpred_list = 0
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
            # if run_labels_data[frame_count] == -1: # For ablation study
            #     break
            mask, processed_boxes, dboxes = get_masks(detect(frame, det_model), img_size, args.mask_method, args.mask_prior, viz)
            dmask = mask[0,0,:,:]
            dmask = cv2.resize(dmask, (width, height))
            dmask = dmask.astype(np.uint8) * 255  # Multiply by 255 to convert bool to uint8
            dmask = np.expand_dims(dmask, axis=-1)
            dmask = np.repeat(dmask, 3, axis=-1)  # Repeat the dimension three times
            dmask = cv2.putText(dmask, f"Attention Masks", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Display Bounding boxes overlayed on current frame
            frame_with_dboxes = frame.copy()
            for box1 in dboxes:
                x, y, x2, y2 = box1
                cv2.rectangle(frame_with_dboxes, (x, y), (x2, y2), (0, 255, 0), 2)
            frame_with_dboxes = cv2.putText(frame_with_dboxes, f"Actual Detections", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Display processed_Bboxes overlayed on current frame
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
                predictions, probs, tpred = test(masks, pred_model, return_probs=True)
                probs_list.append(probs[0])
                preds_list.append(predictions[0])
                # label_list.append(run_labels_data[frame_count]) # for ablation study

                masks = masks[:,:,1:,:,:]
                counter += 1
                neram = time.time() - t2  # neram = time
                total_neram += neram
                neram_list.append(neram)
                tpred_list += tpred
                print(f"Sequence no.: {counter}")
                if counter == 1:
                    first_seq = time.time() - t1
                    print(f"Time for the first sequence (8 masks + 1 prediction): {first_seq:.4} s")
                else:
                    print(f"Time for seq (1 det + 1 pred): {neram:.4} s")
                # Visualize the prediction for each frame
                frame = cv2.putText(frame, f"Prediction: {predictions[0]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                combined_frame = np.hstack((np.vstack((frame, frame_with_processed_boxes)),np.vstack((frame_with_dboxes, dmask))))

                # # Create the plot
                # plt.plot(probs_list, color='blue', label='Probability')
                # plt.plot(neram_list, color='red', label='Seq. Time')
                # plt.plot(tpred_list, color='green', label='Prediction time')
                # plt.xlabel('Time step')
                # plt.ylabel('Value')
                # plt.title('Risk, Time, and Tpred over Time')
                # # plt.ylim(0, 1)
                # plt.legend()
                # plot = plt.gcf()
                # plot.canvas.draw()
                # plot_array = np.array(plot.canvas.renderer.buffer_rgba())

                # cv2.imshow('Plot', plot_array)
                # plt.close()
                cv2.imshow('Visualizer', combined_frame)

                # Write the combined frame to the video file
                # video_writer.write(combined_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        if frame_count == 1000: #for testing camera
            break
        frame_count += 1

    # bal_acc, precision, recall, fscore = get_classification_metrics(preds_list, label_list) # for ablation study
    
    # print(f"Prediction: {preds_list} length: {len(preds_list)} \nLabels: {label_list} length: {len(label_list)}"
        #   f"\n\nBalanced Accuracy: {bal_acc:.4} \nPrecision: {precision:.4} \nRecall: {recall:.4} \nF1 Score: {fscore:.4}\n") # for ablation study
    print(f"\nVideo Duration: {video_duration} s \nTotal processing time: {time.time() - t1:.4} s"
          f"\nTime for first seq (8 detections + 1 prediction): {first_seq:.4} s" 
          f"\nAverage sequence time: {total_neram / counter:.4} s" 
          f"\nAverage prediction time: {tpred_list / counter:.4} s")
    cap.release()
    cv2.destroyAllWindows()
    # video_writer.release()