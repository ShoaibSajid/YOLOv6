import os
import cv2
import glob
# import matplotlib.pyplot as plt

# Define directories
pred_dir = '../runs/inference/yolov6s/labels'
gt_dir = './connector_org/labels/val'
img_dir = './connector_org/images/val'
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),"Comparison")  # Directory of the script
print(f"Saving images in {output_dir}")

# Define class names
class_names = ['g', 'G', 'NG']

# Define colors for different classes
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Red for 'g', Green for 'G', Blue for 'NG'

# Function to read bounding boxes from a label file
def read_bboxes(label_file):
    bboxes = []
    with open(label_file, 'r') as file:
        for line in file:
            parts = line.strip().split()
            class_id = int(parts[0])
            bbox = list(map(float, parts[1:]))
            bboxes.append((class_id, bbox))
    return bboxes

# Function to draw bounding boxes on an image
def draw_bboxes(img, bboxes, color):
    h, w = img.shape[:2]
    for class_id, bbox in bboxes:
        try: x_center, y_center, width, height = bbox
        except: x_center, y_center, width, height, _ = bbox
        x1 = int((x_center - width / 2) * w)
        y1 = int((y_center - height / 2) * h)
        x2 = int((x_center + width / 2) * w)
        y2 = int((y_center + height / 2) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), color[class_id % len(color)], 2)
        cv2.putText(img, class_names[class_id], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color[class_id % len(color)], 2)

# Get list of image files
img_files = glob.glob(os.path.join(img_dir, '*.png'))

for img_file in img_files:
    img_name = os.path.basename(img_file)
    img = cv2.imread(img_file)
    pred_file = os.path.join(pred_dir, img_name.replace('.png', '.txt'))
    gt_file = os.path.join(gt_dir, img_name.replace('.png', '.txt'))
    
    count_drawn = 0

    if os.path.exists(pred_file):
        pred_bboxes = read_bboxes(pred_file)
        # draw_bboxes(img, pred_bboxes, colors)
    
    if os.path.exists(gt_file):
        gt_bboxes = read_bboxes(gt_file)
        # draw_bboxes(img, gt_bboxes, colors)

    # Create dictionaries to store bounding boxes for predictions and ground truth
    pred_dict = {i: [] for i in range(len(colors))}
    gt_dict = {i: [] for i in range(len(colors))}
    
    for class_id, bbox in pred_bboxes:
        pred_dict[class_id].append(bbox)
    for class_id, bbox in gt_bboxes:
        gt_dict[class_id].append(bbox)
        

    # Draw unmatched bounding boxes
    for class_id in range(len(colors)):
        for pred_bbox in pred_dict[class_id]:
            matched = False
            for gt_bbox in gt_dict[class_id]:
                # Compare bounding boxes for each class
                if abs(pred_bbox[0] - gt_bbox[0]) < 0.0015 and abs(pred_bbox[1] - gt_bbox[1]) < 0.0015:
                    matched = True
                    break
            if not matched:
                # Draw bounding box if not matched
                draw_bboxes(img, [(class_id, gt_bbox)], [colors[class_id]])
                draw_bboxes(img, [(class_id, pred_bbox)], [colors[class_id]])
                count_drawn +=1


    # Add image file name on top
    cv2.putText(img, img_name, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Count predictions and ground truth for each class
    pred_count = {i: 0 for i in range(len(colors))}
    gt_count = {i: 0 for i in range(len(colors))}
    for class_id, _ in pred_bboxes:
        pred_count[class_id] += 1
    for class_id, _ in gt_bboxes:
        gt_count[class_id] += 1

    # Display counts on the image
    y_offset = 80
    for i in range(len(colors)):
        cv2.putText(img, f'Pred Class {i}: {pred_count[i]}', (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(img, f'GT Class {i}: {gt_count[i]}', (20, y_offset + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_offset += 40

    # Check if counts do not match for any class
    counts_match = all(pred_count[i] == gt_count[i] for i in range(len(colors)))

    # Save the annotated image only if counts do not match
    if not counts_match and count_drawn:
        # Save the output image in the same folder as the code
        output_img_path = os.path.join(output_dir, f'annotated_{img_name}')
        cv2.imwrite(output_img_path, img)
        print(f'Saved annotated image: {output_img_path}')
