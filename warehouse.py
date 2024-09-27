import cv2
import time
import numpy as np
import torch
import os

from dataset.Realtimetest.real_inference_dataset import RealInferenceDataset
from dataset.Realtimetest.Detector import detect
from TVT.test import test
from tRTfiles.yolo_tensorrt_engine import load_engine, allocate_buffers
from TVT.utils import *

import matplotlib.pyplot as plt

class Warehouse:
    # def __init__(self, pred_context, yolo_context, yolo_tensorrt, pred_tensorrt, args, img_size, video_path, labels_path, preds_list, label_list):    # FOR ABLATION STUDY
    def __init__(self,args, img_size, video_path):
        # self.pred_model = pred_model
        # self.det_model = det_model # object detection model on CUDA
        self.args = args
        self.img_size = img_size
        self.video_path = video_path
        self.return_probs = False

        # Load the TensorRT Detection engine
        yolo_engine = load_engine("/home/tue/risknet/Realtime-RiskNET/dataset/Realtimetest/yolov10s.engine")
        pred_engine = load_engine("/home/tue/risknet/Realtime-RiskNET/pred_models/old/pred_model.trt")
        self.pred_context = pred_engine.create_execution_context()
        self.yolo_context = yolo_engine.create_execution_context()
        self.yolo_tensorrt = allocate_buffers(yolo_engine) #inputs, outputs, bindings, stream
        self.pred_tensorrt = allocate_buffers(pred_engine)

        # Get dimensions of input frames
        self.height, self.width = 360, 480

        # FOR ABLATION STUDY
        # self.pred_context = pred_context
        # self.yolo_context = yolo_context
        # self.yolo_tensorrt = yolo_tensorrt #inputs, outputs, bindings, stream
        # self.pred_tensorrt = pred_tensorrt
        # self.labels_path = labels_path
        # self.preds_list = preds_list
        # self.label_list = label_list

    def detect(self): # object detection model
        # return detect(self.frame, self.det_model)
        return detect(self.frame, self.yolo_context, self.yolo_tensorrt) # TensorRT
    
    def get_masks(self): # Creates attention masks from detections
        instance = RealInferenceDataset(self.detect(), self.img_size, self.args.mask_method, self.args.mask_prior, self.args.viz)
        return instance.get_masks()
    
    def test(self, masks): # Risk prediction model (inference)
        # return test(masks, self.pred_model, return_probs=self.return_probs)
        return test(masks, self.pred_context, self.pred_tensorrt, return_probs=self.return_probs) # TensorRT
        
    def append_detections_masks(self):

        # Cold start the models
        print('-'*79 + "\nCold starting the models...")
        self.frame = np.ones((self.height, self.width, 3))
        cold_masks = self.get_masks()
        cold_masks = torch.tensor(cold_masks).unsqueeze(0).float()
        cold_masks = cold_masks.repeat(1, 1, 8, 1, 1)
        self.test(cold_masks)

        # Start the camera/video
        if isinstance(self.video_path, int):
            print('-'*79 + "\nStreaming Camera...")
        else:
            print('-'*79 + "\nStreaming Video...")
        
        cap = cv2.VideoCapture(self.video_path) # 0 for webcam

        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Stream fps: {fps}")
        print(f"Predictions on 10 fps") if self.args.tenfps else None
        print(f"Stream resolution: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)} x {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")

        multiple = int(fps / 10) # 10 is the frame rate of the model

        if self.video_path == 0 or self.video_path == 1:
            video_duration = 'Live streaming'
        else:
            video_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps

        # FOR ABLATION STUDY
        # run_labels_data = np.loadtxt(self.labels_path, delimiter=',')
        # run_labels_data[run_labels_data == 1] = 0
        # run_labels_data[run_labels_data == 2] = 1

        frame_count = 0
        total_neram = 0
        tpred = 0
        first = True
        counter = 0
        pred_list = []
        t1 = time.time()

        while True:
            ret, self.frame = cap.read()
            t2 = time.time()
            if not ret:
                break
            fps_flag = not self.args.tenfps or (self.args.tenfps and (frame_count == 0 or frame_count % multiple == 0))
            if fps_flag:

                # if run_labels_data[frame_count] == -1: # FOR ABLATION STUDY
                #         break
                
                #if frame is not 480x360, resize it
                if self.frame.shape[0] != self.height or self.frame.shape[1] != self.width:
                    self.frame = cv2.resize(self.frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
                mask = self.get_masks()
                mask = torch.tensor(mask).unsqueeze(0).float()
                if first:
                    masks = mask
                    first = False
                else:
                    masks = torch.cat((masks, mask), dim=2)

                if masks.shape[2] == 8:
                    predictions, t_total = self.test(masks)
                    # self.preds_list.append(predictions[0]) # FOR ABLATION STUDY
                    # self.label_list.append(run_labels_data[frame_count]) # FOR ABLATION STUDY
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

            # Doesn't have to be inside the drop frames if statement          
            if isinstance(self.video_path, int) and frame_count == 500: #for testing camera
                break

            frame_count += 1

        print(f"\nVideo Duration: {video_duration} s"
              f"\nTotal processing time: {time.time() - t1:.4} s"
              f"\nTime for first seq (8 detections + 1 prediction): {first_seq:.4} s" 
              f"\nAverage sequence time: {total_neram / counter:.4} s" 
              f"\nAverage prediction time: {tpred / counter:.4} s")
        cap.release()

    def append_detections_masks_viz(self):

        # Cold start the models
        print('-'*79 + "\nCold starting the models...")
        self.frame = np.ones((self.height, self.width, 3))
        cold_masks,_,_ = self.get_masks()
        print(f"Size of cold_masks: {cold_masks.shape}")
        cold_masks = torch.tensor(cold_masks).unsqueeze(0).float()
        cold_masks = cold_masks.repeat(1, 1, 8, 1, 1)
        self.test(cold_masks)

        # Start the camera/video
        if isinstance(self.video_path, int):
            print('-'*79 + "\nStreaming Camera...")
        else:
            print('-'*79 + "\nStreaming Video...")
        
        cap = cv2.VideoCapture(self.video_path) # 0 for webcam

        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Stream fps: {fps}")
        print(f"Predictions on 10 fps") if self.args.tenfps else None
        print(f"Stream resolution: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)} x {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")

        multiple = int(fps / 10) # 10 is the frame rate of the model

        if isinstance(self.video_path, int):
            video_duration = "Live Streaming"
        else:
            video_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps

        # # FOR ABLATION STUDY
        # run_labels_data = np.loadtxt(self.labels_path, delimiter=',')
        # run_labels_data[run_labels_data == 1] = 0
        # run_labels_data[run_labels_data == 2] = 1

        if self.args.graph:
            probs_list = []
            neram_list = []
            tpred_list = []
            self.return_probs = True

        frame_count = 0
        total_neram = 0
        first = True
        counter = 0
        total_tpred = 0
        t1 = time.time()
        while True:
            t2 = time.time()
            ret, self.frame = cap.read()
            if not ret:
                break
            fps_flag = not self.args.tenfps or (self.args.tenfps and (frame_count == 0 or frame_count % multiple == 0))
            if fps_flag:
                # if frame is not 480x360, resize the frame to the required size
                if self.frame.shape[0] != self.height or self.frame.shape[1] != self.width:
                    self.frame = cv2.resize(self.frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
                # if run_labels_data[frame_count] == -1: # FOR ABLATION STUDY
                #     break
                mask, processed_boxes, dboxes = self.get_masks()
                dmask = mask[0,0,:,:]
                mask = torch.tensor(mask).unsqueeze(0).float()

                if first:
                    masks = mask
                    first = False
                else:
                    masks = torch.cat((masks, mask), dim=2)

                if masks.shape[2] == 8:
                    if self.args.graph:
                        predictions, probs, tpred = self.test(masks)
                        probs_list.append(probs[0])
                    else:
                        predictions, tpred = self.test(masks)
                    
                    # preds_list.append(predictions[0]) # FOR ABLATION STUDY
                    # label_list.append(run_labels_data[frame_count]) # FOR ABLATION STUDY
                    print(f"Prediction: {predictions[0]}")
                    masks = masks[:,:,1:,:,:]
                    counter += 1
                    neram = time.time() - t2  # neram = time
                    total_neram += neram                        
                    total_tpred += tpred
                    print(f"Sequence no.: {counter}")
                    if counter == 1:
                        first_seq = time.time() - t1
                        print(f"Time for the first sequence (8 masks + 1 prediction): {first_seq:.4} s")
                    else:
                        print(f"Time for seq (1 det + 1 pred): {neram:.4} s")

                    #### Visualize the prediction for each frame ####
                    dmask = cv2.resize(dmask, (self.width, self.height))
                    dmask = dmask.astype(np.uint8) * 255    # Multiply by 255 to convert bool to uint8
                    dmask = np.expand_dims(dmask, axis=-1)
                    dmask = np.repeat(dmask, 3, axis=-1)    # Repeat the dimension three times
                    dmask = cv2.putText(dmask, f"Attention Masks", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                    # Display Bounding boxes overlayed on current frame
                    frame_with_dboxes = self.frame.copy()
                    for box1 in dboxes:
                        x1, y1, x2, y2 = box1
                        cv2.rectangle(frame_with_dboxes, (x1, y1), (x2, y2), (0, 255, 0), 2) if predictions[0] == 0 else cv2.rectangle(frame_with_dboxes, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    frame_with_dboxes = cv2.putText(frame_with_dboxes, f"Actual Detections", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    # Display processed_Bboxes overlayed on current frame
                    frame_with_processed_boxes = self.frame.copy()
                    for box2 in processed_boxes:
                        x1, y1, x2, y2 = box2
                        cv2.rectangle(frame_with_processed_boxes, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    frame_with_processed_boxes = cv2.putText(frame_with_processed_boxes, f"Filtered Detections", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    self.frame = cv2.putText(self.frame, f"Prediction: {predictions[0]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    combined_frame = np.hstack((np.vstack((self.frame, frame_with_processed_boxes)),np.vstack((frame_with_dboxes, dmask))))
                    
                    # Create the plot
                    if self.args.graph:
                        neram_list.append(neram)
                        tpred_list.append(tpred)
                        plt.plot(probs_list, color='blue', label='Probability')
                        plt.plot(neram_list, color='red', label='Seq. Time')
                        plt.plot(tpred_list, color='green', label='Prediction time')
                        plt.xlabel('Time step')
                        plt.ylabel('Value')
                        plt.title('Risk, Time, and Tpred over Time')
                        # plt.ylim(0, 1)
                        plt.legend()
                        plot = plt.gcf()
                        plot.canvas.draw()
                        plot_array = np.array(plot.canvas.renderer.buffer_rgba())
                        cv2.imshow('Plot', plot_array)
                        plt.close()

                    cv2.imshow('Visualizer', combined_frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
            # Doesn't have to be inside the drop frames if statement          
            if isinstance(self.video_path, int) and frame_count == 500: #for testing camera
                break
            frame_count += 1

        # bal_acc, precision, recall, fscore = get_classification_metrics(preds_list, label_list) # FOR ABLATION STUDY
    
        # print(f"Prediction: {preds_list} length: {len(preds_list)} \nLabels: {label_list} length: {len(label_list)}"
            #   f"\n\nBalanced Accuracy: {bal_acc:.4} \nPrecision: {precision:.4} \nRecall: {recall:.4} \nF1 Score: {fscore:.4}\n") # FOR ABLATION STUDY
    
        print(f"\nFrame count: {frame_count}"
              f"\nVideo Duration: {video_duration} s"
              f"\nTotal processing time: {time.time() - t1:.4} s"
              f"\nTime for first seq (8 detections + 1 prediction): {first_seq:.4} s" 
              f"\nAverage sequence time: {total_neram / counter:.4} s" 
              f"\nAverage prediction time: {total_tpred / counter:.4} s")
        cap.release()
        cv2.destroyAllWindows()
