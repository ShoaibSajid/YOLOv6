import cv2
import torch
from pathlib import Path
import sys
import time

# Assuming YOLOv6 is in the current directory and `yolov6` is the module name
sys.path.append('../')
# from yolov6.utils.events import load_checkpoint

from yolov6.utils.checkpoint import load_checkpoint
from yolov6.data.data_augment import letterbox
from yolov6.utils.nms import non_max_suppression


# Load YOLOv6 model
model_path = './working_weights.pt'  # Update with your model path
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.load(model_path, map_location=device)
_model = model['model']
_model.float()
_model.eval()

# Class names (for COCO dataset)
class_names = ["g","G","NG"]

# Set the target class
target_class = 'NG'
target_class_id = class_names.index(target_class)

# def draw_boxes(frame, pred, conf_threshold=0.25):
#     if pred[0] is not None:
#         for det in pred[0]:
#             if det[4].item() > conf_threshold:
#                 x1, y1, x2, y2 = det[:4]
#                 x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

#                 # Scale coordinates back to the original frame size
#                 frame_height, frame_width = frame.shape[:2]
#                 x1 = int(x1 * frame_width / 640)
#                 y1 = int(y1 * frame_height / 640)
#                 x2 = int(x2 * frame_width / 640)
#                 y2 = int(y2 * frame_height / 640)

#                 # Draw the bounding box
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

#                 # Draw the label
#                 label = f"{class_names[int(det[5])]} {det[4]:.2f}"
#                 cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def draw_boxes(frame, pred, frame_width, frame_height, conf_threshold=0.5):
    if pred[0] is not None:
        for det in pred[0]:
            if det[4].item() > conf_threshold:
                x1, y1, x2, y2 = det[:4]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Scale coordinates back to the original frame size
                x1 = int(x1 * frame_width / 640)
                y1 = int(y1 * frame_height / 640)
                x2 = int(x2 * frame_width / 640)
                y2 = int(y2 * frame_height / 640)

                # Draw the bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw the label
                label = f"{class_names[int(det[5])]} {det[4]:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


def process_frame(frame, model, device):
    # Preprocess the frame
    img = letterbox(frame, new_shape=(640, 640), stride=32)[0]  # Resize with padding
    img = img.transpose((2, 0, 1))  # HWC to CHW
    img = torch.from_numpy(img).to(device).float() / 255.0  # Normalize
    img = img.unsqueeze(0)  # Add batch dimension

    # Inference
    with torch.no_grad():
        pred = model(img)[0]

    # Apply NMS
    pred = non_max_suppression(pred, 0.4, 0.45, classes=None, agnostic=False)
    
    # gn = torch.tensor(img_src.shape)[[1, 0, 1, 0]]  # normalization gain whwh
    
    return pred

def count_instances(pred, target_class_id, conf_threshold=0.25):
    count = 0
    if pred[0] is not None:
        for det in pred[0]:  # Since pred is a list of detections per image
            if det[4].item() > conf_threshold and int(det[5]) == target_class_id:
                count += 1
    return count

# Video processing
video_path = 'compressed_connector.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Output video writer
out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

i=0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    i+=1
    
    if not i%10==0 or i<500: continue
    
    start_time = time.time()
    
    crop_size=(400,900)
 
    # Get the dimensions of the frame
    height, width, _ = frame.shape

    # Calculate the coordinates for the central crop
    center_x, center_y = width // 2, height // 2
    crop_width, crop_height = crop_size

    start_x = max(center_x - crop_width // 2, 0)
    start_y = max(center_y - crop_height // 2, 0)
    end_x = min(center_x + crop_width // 2, width)
    end_y = min(center_y + crop_height // 2, height)

    # Crop the central part of the frame
    cropped_frame = frame[start_y:end_y, start_x:end_x]

    # Process the frame and get predictions
    pred = process_frame(cropped_frame, _model, device)

    # Count instances of the target class
    instance_count = count_instances(pred, target_class_id)

    # Draw the predicted bounding boxes on the frame
    # draw_boxes(cropped_frame, pred)
    draw_boxes(cropped_frame, pred, cropped_frame.shape[1], cropped_frame.shape[0])

    
    # Draw the instance count on the frame
    cv2.putText(cropped_frame, f"{target_class}: {instance_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Write the processed frame to the output video
    out.write(cropped_frame)

    # Display the frame (optional)
    # resize_factor = 0.3
    # cv2.imshow('Frame', cv2.resize(cropped_frame, (0,0), fx=resize_factor, fy=resize_factor) )
    cv2.imshow('Frame', cropped_frame )
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    
    cropped_frame, pred = [], []
    end_time = time.time()
    print(f"Processed frame in {end_time - start_time:.2f} seconds. {target_class} count: {instance_count}")
    
# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
