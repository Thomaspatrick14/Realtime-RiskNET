import cv2
import threading
from pathlib import Path
from Detector import detect
import os
import time
import torchvision
import torch

def capture_frames():
    print(f"Loading the model")
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    if torch.cuda.is_available():
        print("CUDA available: loading the model on the GPU")
        model = model.cuda()
    print(f"Model loaded")
    model.eval()

    folder_path = "C:/Users/up650/Downloads/New folder (2)"
    video_path = os.path.join(folder_path, "output.mp4")  # Replace with the path to your video file
    cap = cv2.VideoCapture(video_path)

    # json_save_path = Path(folder_path, f"frames.json")
    # if json_save_path.exists():
    #     os.remove(json_save_path)
    
    frame_count = 0
    t = time.time()

    ## For running the detector on every frame 3rd frame
    # while True:
    #     ret, frame = cap.read()
    #     if not ret:
    #         break

    #     if frame_count == 0 or frame_count % 3 == 0:
    #         # Process every 3rd frame
    #         detect(frame, frame_count, folder_path, model)
        
    #     # cv2.imshow('Original Frame', frame)
    #     if frame_count == 21:
    #         break
    #     frame_count += 1
    #     # if cv2.waitKey(1) == 27:  # Check if the Esc key is pressed
    #     #     break

    # cap.release()
    # print(f"Total time taken: {(time.time() - t):.3f} seconds")
    # # cv2.destroyAllWindows()

    frames_to_process = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count == 0 or frame_count % 3 == 0:
            frames_to_process.append(frame)

        # if frame_count == 21:
        #     break
        frame_count += 1
    
    detections = detect(frames_to_process, folder_path, model)
    cap.release()
    print(f"Total time taken: {(time.time() - t):.3f} seconds")

# Start a thread for capturing frames
capture_thread = threading.Thread(target=capture_frames)
capture_thread.start()
