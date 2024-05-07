import torch
import numpy as np

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
if torch.cuda.is_available():
        print("CUDA available: loading the detection model on the GPU")
        model = model.cuda()
model.eval()

# Images
imgs = ['C:/JP/TUe/2nd year Internship and thesis/Thesis/Code/Idiada/car_to_bicycle_turning/2023-01-23-16-55-30_filtered_cropped/1674489347659700796.png']  # batch of images
print(f'type(imgs): {type(imgs)}')
# Inference
results = model(imgs)

# Results
# results.print()
results.show()  # or .show()

# results.xyxy[0]  # img1 predictions (tensor)
# print(int(results.pred[0][1][-1].item()))  # img1 predictions (box xyxy, conf, cls)
# length = len(results.pred[0])
# for i in range(length):
#     print(f'\Predicted classes: {results.names[int(results.pred[0][i][-1].item())]}')  # img1 classes
#     print(f'Predicted bounding box: {results.pred[0][i][:4].tolist()}')  # img1 xyxy
#     print(f'Predicted bounding box confidence: {results.pred[0][i][4].item():.2f}')  # img1 conf
#     print('-------------------------------------\n')
# print(f'Predicted classes: {results.names[int(results.pred[0][1][-1].item())]}')  # img1 classes
pred_bboxes = results.pred[0][:, :4].detach().cpu().numpy()
pred_scores = results.pred[0][:, 4].detach().cpu().numpy()
pred_classes = results.pred[0][:, 5].detach().cpu().numpy().astype(int)
print(f'Predicted classes: {int(results.pred[0][1][-1].item())}')
print(f'predicted bounding box: {pred_bboxes}')  # img1 xyxy
print(f'predicted bbox conf: {pred_scores}')
print(f'predicted classes: {pred_classes}')  # img1 classes
boxes = pred_bboxes[pred_scores >= 0.5].astype(np.int32)
classes = pred_classes[pred_scores >= 0.5].astype(np.int32)
print(f'\nboxes: {boxes}')
print(f'classes: {classes}')
# print(f'Predicted bounding box confidence: {results.pred[0][0][4].item():.2f}')