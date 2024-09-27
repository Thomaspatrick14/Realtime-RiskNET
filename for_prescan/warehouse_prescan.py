import cv2
import time
import numpy as np
import torch
import os
import socket
import struct
from sklearn.metrics import confusion_matrix

from dataset.Realtimetest.real_inference_dataset import RealInferenceDataset
from dataset.Realtimetest.Detector import detect
from TVT.test import test
from tRTfiles.yolo_tensorrt_engine import load_engine, allocate_buffers
from TVT.utils import *

import matplotlib.pyplot as plt

class Warehouse:
    def __init__(self, args, img_size, video_path):
        # self.pred_model = pred_model
        # self.det_model = det_model # object detection model on CUDA
        self.args = args
        self.img_size = img_size
        self.return_probs = False

        # Load the TensorRT Detection engine
        yolo_engine = load_engine("/home/tue/risknet/Realtime-RiskNET/dataset/Realtimetest/yolov10s.engine")
        pred_engine = load_engine("/home/tue/risknet/Realtime-RiskNET/pred_models/model_engine_fp16.trt")
        self.pred_context = pred_engine.create_execution_context()
        self.yolo_context = yolo_engine.create_execution_context()
        self.yolo_tensorrt = allocate_buffers(yolo_engine) #inputs, outputs, bindings, stream
        self.pred_tensorrt = allocate_buffers(pred_engine)

        # Get dimensions of input frames
        self.height, self.width = 360, 480

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

        print(f"Predictions on 10 fps") if self.args.tenfps else None

        multiple = int(3) # 10 is the frame rate of the model
        video_duration = "Live Streaming"
        
        ############################################################
        # Load the ground truth from CSV file (for prescan metrics)
        ############################################################
        labels_path = "/home/tue/risknet/Realtime-RiskNET/for_prescan/For_prescan/Demo_mostImportant_labels.csv"  # Path to your CSV file containing the ground truths
        targets_list = []
        with open(labels_path, newline='') as csvfile:
            ground_truths = csv.reader(csvfile, delimiter=',')
            targets_list = [int(row[0]) for row in ground_truths]  # Assuming ground truth values are in the first column

        ############################################################
                    # TCP/IP & UDP socket creation #
        ############################################################

        # Define image dimensions
        num_channels = 3
        frame_size = self.height * self.width * num_channels  # Total number of bytes in the image
        single_layer = self.height * self.width  # Number of bytes in a single layer

        # Create a TCP/IP socket for receiving images
        tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        tcp_socket.bind(('0.0.0.0', 5005))  # Bind to the listening port for TCP
        tcp_socket.listen(1)

        print("Waiting for connection...")
        connection, _ = tcp_socket.accept()
        print("Connected")

        # Create a UDP socket for sending data back
        udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # Target IP and port for sending data
        target_ip = '192.168.0.1'  # IP address of the other computer running Simulink
        target_port = 5006  # UDP port for sending data

        frame_data = bytearray() # Buffer to accumulate frame packets
        #############################################################

        frame_count = 0
        total_neram = 0
        tpred = 0
        first = True
        counter = 0
        pred_list = []      # for prescan metrics
        t1 = time.time()

        try:
            while True:
                                
                while len(frame_data) < frame_size: # Accumulate frame packets until the frame is complete
                    packet = connection.recv(frame_size - len(frame_data))  # Receive TCP packet
                    if not packet:
                        break
                    frame_data.extend(packet)

                if len(frame_data) == frame_size:
                    t2 = time.time()
                    # Convert the byte data to a NumPy array with dtype uint8 and reshape to (height, width, channels)
                    self.frame = np.stack((np.frombuffer(frame_data, dtype=np.uint8)[single_layer*2:].reshape((self.width, self.height)).transpose(),
                                    np.frombuffer(frame_data, dtype=np.uint8)[single_layer:single_layer*2].reshape((self.width, self.height)).transpose(),
                                    np.frombuffer(frame_data, dtype=np.uint8)[:single_layer].reshape((self.width, self.height)).transpose()), axis=-1)
                    frame_data.clear() # Clear the buffer for the next frame
                    
                    # cv2.imshow('Visualizer', self.frame)
                    fps_flag = not self.args.tenfps or (self.args.tenfps and (frame_count == 0 or frame_count % multiple == 0))
                    if fps_flag:

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
                            tpred += t_total
                            pred_list.append(predictions[0])    # for prescan metrics
                            print(f"Prediction: {predictions[0]}")
                            udp_socket.sendto(np.uint8(predictions[0]).tobytes(), (target_ip, target_port)) # Send the prediction via UDP
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
                            
                            # cv2.imshow('Visualizer', self.frame)
                    frame_count += 1

                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #         break

        finally:    
            # ##### for prescan metrics #####
            print(f"len(pred_list): {len(pred_list)}")
            bal_acc, precision, recall, fscore = get_classification_metrics(pred_list, targets_list)
            print(f"\nPrecision: {precision}\nF-score: {fscore}\nBalanced Accuracy: {bal_acc}\nRecall: {recall}")

            # Compute confusion matrix
            confusion_mat = confusion_matrix(targets_list, pred_list)
            print("\n Confusion Matrix:\n")
            print(confusion_mat)
            print("\nLegend \n[TN  FP]\n[FN  TP]\n 0 = TN + FP \n 1 = FN + TP\n")
            
            #################################
                
            print(f"\nFrame count: {frame_count}"
                f"\nVideo Duration: {video_duration} s"
                f"\nTotal processing time: {time.time() - t1:.4} s"
                f"\nTime for first seq (8 detections + 1 prediction): {first_seq:.4} s" 
                f"\nAverage sequence time: {total_neram / counter:.4} s" 
                f"\nAverage prediction time: {tpred / counter:.4} s")
            connection.close()
            tcp_socket.close()
            udp_socket.close()
            cv2.destroyAllWindows()

    def append_detections_masks_viz(self):

        # Cold start the models
        print('-'*79 + "\nCold starting the models...")
        self.frame = np.ones((self.height, self.width, 3))
        cold_masks,_,_ = self.get_masks()
        print(f"Size of cold_masks: {cold_masks.shape}")
        cold_masks = torch.tensor(cold_masks).unsqueeze(0).float()
        cold_masks = cold_masks.repeat(1, 1, 8, 1, 1)
        self.test(cold_masks)

        print(f"Predictions on 10 fps") if self.args.tenfps else None

        multiple = int(3) # 10 is the frame rate of the model
        video_duration = "Live Streaming"

        # # For ablation study
        # run_labels_data = np.loadtxt(self.labels_path, delimiter=',')
        # run_labels_data[run_labels_data == 1] = 0
        # run_labels_data[run_labels_data == 2] = 1

        if self.args.graph:
            probs_list = []
            neram_list = []
            tpred_list = []
            self.return_probs = True

        ############################################################
        # Load the ground truth from CSV file
        ############################################################
        # labels_path = "/home/tue/Downloads/EuroNCAP_bicycle_labels.csv"  # Path to your CSV file containing the ground truths
        # targets_list = []
        # with open(labels_path, newline='') as csvfile:
        #     ground_truths = csv.reader(csvfile, delimiter=',')
        #     targets_list = [int(row[0]) for row in ground_truths]  # Assuming ground truth values are in the first column

        ############################################################
                    # TCP/IP & UDP socket creation #
        ############################################################
        # Define image dimensions
        num_channels = 3
        frame_size = self.height * self.width * num_channels  # Total number of bytes in the image
        single_layer = self.height * self.width  # Number of bytes in a single layer

        # Create a TCP/IP socket for receiving images
        tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        tcp_socket.bind(('0.0.0.0', 5005))  # Bind to the listening port for TCP
        tcp_socket.listen(1)

        print("Waiting for connection...")
        connection, _ = tcp_socket.accept()
        print("Connected")

        # Create a UDP socket for sending data back
        udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # Target IP and port for sending data
        target_ip = '192.168.0.1'  # IP address of the other computer running Simulink
        target_port = 5006  # UDP port for sending data

        frame_data = bytearray() # Buffer to accumulate frame packets
        #############################################################

        frame_count = 0
        total_neram = 0
        pred_list = []
        first = True
        counter = 0
        total_tpred = 0
        t1 = time.time()

        try:
            while True:
                t2 = time.time()

                while len(frame_data) < frame_size: # Accumulate frame packets until the frame is complete
                    packet = connection.recv(frame_size - len(frame_data))  # Receive TCP packet
                    if not packet:
                        break
                    frame_data.extend(packet)

                if len(frame_data) == frame_size:
                    # Convert the byte data to a NumPy array with dtype uint8 and reshape to (height, width, channels)
                    self.frame = np.stack((np.frombuffer(frame_data, dtype=np.uint8)[single_layer*2:].reshape((self.width, self.height)).transpose(),
                                    np.frombuffer(frame_data, dtype=np.uint8)[single_layer:single_layer*2].reshape((self.width, self.height)).transpose(),
                                    np.frombuffer(frame_data, dtype=np.uint8)[:single_layer].reshape((self.width, self.height)).transpose()), axis=-1)
                    frame_data.clear() # Clear the buffer for the next frame

                    fps_flag = not self.args.tenfps or (self.args.tenfps and (frame_count == 0 or frame_count % multiple == 0))
                    if fps_flag:
                        # if frame is not 480x360, resize the frame to the required size
                        if self.frame.shape[0] != self.height or self.frame.shape[1] != self.width:
                            self.frame = cv2.resize(self.frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
                        # if run_labels_data[frame_count] == -1: # For ablation study
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

                            pred_list.append(predictions[0])
                            # preds_list.append(predictions[0]) # for ablation study
                            # label_list.append(run_labels_data[frame_count]) # for ablation study
                            print(f"Prediction: {predictions[0]}")
                            udp_socket.sendto(np.uint8(predictions[0]).tobytes(), (target_ip, target_port)) # Send the prediction via UDP
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

                            self.frame = cv2.putText(self.frame.copy(), f"Prediction: {predictions[0]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
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
                # if isinstance(self.video_path, int) and frame_count == 500: #for testing camera
                #     break
                frame_count += 1
        finally:
            # bal_acc, precision, recall, fscore = get_classification_metrics(pred_list[1:], targets_list[7:])
            # print(f"\nBalanced Accuracy: {bal_acc}\nPrecision: {precision}\nRecall: {recall}\nF-score: {fscore}")

            # # Compute confusion matrix
            # confusion_mat = confusion_matrix(targets_list[7:], pred_list[1:])
            # print("\n Confusion Matrix:\n")
            # print(confusion_mat)
            # print("\nLegend \n[TN  FP]\n[FN  TP]\n 0 = TN + FP \n 1 = FN + TP\n")
            # Save pred_list as a CSV file
            pred_list_path = "/home/tue/Downloads/For_prescan/pred_list.csv"  # Path to save the predictions
            with open(pred_list_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Prediction"])  # Write header
                for pred in pred_list:
                    writer.writerow([pred])
            print(f"\nFrame count: {frame_count}"
                f"\nVideo Duration: {video_duration} s"
                f"\nTotal processing time: {time.time() - t1:.4} s"
                f"\nTime for first seq (8 detections + 1 prediction): {first_seq:.4} s" 
                f"\nAverage sequence time: {total_neram / counter:.4} s" 
                f"\nAverage prediction time: {total_tpred / counter:.4} s")
            connection.close()
            tcp_socket.close()
            udp_socket.close()
            cv2.destroyAllWindows()