import cv2
import numpy as np
import os
import glob

# Paths to the test images and labels
image_folder = "./CHT_testing/test/images"
label_folder = "./CHT_testing/test/labels"  # YOLO format labels

# Get list of image files
image_files = sorted(glob.glob(os.path.join(image_folder, "*.jpg")))  # Adjust extension if needed

# Variables for IoU and mAP calculation
iou_scores = []
tp = 0  # True positives at IoU >= 0.5
total_images = 0

for image_file in image_files:
    # Read the image
    frame = cv2.imread(image_file)
    if frame is None:
        print(f"Error: Could not read image {image_file}.")
        continue

    height, width = frame.shape[:2]

    # Read the corresponding YOLO label
    base_name = os.path.basename(image_file)
    label_file = os.path.join(label_folder, os.path.splitext(base_name)[0] + ".txt")

    if not os.path.exists(label_file):
        print(f"Warning: Label file {label_file} not found.")
        continue

    # Read YOLO label (assuming single object per image)
    with open(label_file, 'r') as f:
        lines = f.readlines()

    if not lines:
        print(f"Warning: No labels found in {label_file}.")
        continue

    total_images += 1

    # YOLO format: class x_center y_center width height (all normalized between 0 and 1)
    # Assuming the first line contains the bounding box for the golf ball
    yolo_data = lines[0].strip().split()
    class_id, x_center, y_center, bbox_width, bbox_height = map(float, yolo_data)

    # Convert normalized coordinates to absolute pixel values for ground truth
    x_center *= width
    y_center *= height
    bbox_width *= width
    bbox_height *= height

    x1_gt = int(x_center - bbox_width / 2)
    y1_gt = int(y_center - bbox_height / 2)
    x2_gt = int(x_center + bbox_width / 2)
    y2_gt = int(y_center + bbox_height / 2)

    # Draw ground truth rectangle for visualization
    frame_gt = frame.copy()
    cv2.rectangle(frame_gt, (x1_gt, y1_gt), (x2_gt, y2_gt), (255, 0, 0), 2)  # Blue rectangle for ground truth

    # Convert frame to HSV color space for color thresholding
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the color range for white color in HSV
    lower_white = np.array([0, 0, 160])
    upper_white = np.array([180, 30, 255])

    # Create a mask to isolate white regions
    mask_white = cv2.inRange(hsv_frame, lower_white, upper_white)

    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, kernel)
    mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, kernel)

    # Apply the mask to the frame
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask_white)

    # Convert masked frame to grayscale
    gray_frame = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise and improve circle detection
    gray_blurred = cv2.GaussianBlur(gray_frame, (9, 9), 2)

    # Only detect circles in the lower half
    # Zero out the top half of the gray_blurred image
    gray_blurred[:height//2, :] = 0

    # Apply Hough Circle Transform
    circles = cv2.HoughCircles(
        gray_blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=30,
        param1=50,
        param2=15,       # Adjust as needed
        minRadius=5,
        maxRadius=50
    )

    if circles is not None:
        # Round circle parameters and convert to integers
        circles = np.uint16(np.around(circles))

        # Assume the first circle is the ball
        detected_circle = circles[0, 0]

        # Circle parameters
        center_detected = (detected_circle[0], detected_circle[1])
        radius_detected = detected_circle[2]

        # Convert the circle to a bounding box (rectangle)
        x1_det = max(0, center_detected[0] - radius_detected)
        y1_det = max(0, center_detected[1] - radius_detected)
        x2_det = min(width - 1, center_detected[0] + radius_detected)
        y2_det = min(height - 1, center_detected[1] + radius_detected)

        x1_det, y1_det, x2_det, y2_det = int(x1_det), int(y1_det), int(x2_det), int(y2_det)

        # Draw the detected bounding box
        cv2.rectangle(frame_gt, (x1_det, y1_det), (x2_det, y2_det), (0, 255, 0), 4)  # Green rectangle for detection
        cv2.circle(frame_gt, center_detected, radius_detected, (0, 255, 255), 4)  # Green circle around the ball

        # Compute IoU between detected bbox and ground truth bbox
        # Intersection
        inter_x1 = max(x1_gt, x1_det)
        inter_y1 = max(y1_gt, y1_det)
        inter_x2 = min(x2_gt, x2_det)
        inter_y2 = min(y2_gt, y2_det)

        inter_width = max(0, inter_x2 - inter_x1 + 1)
        inter_height = max(0, inter_y2 - inter_y1 + 1)
        intersection = inter_width * inter_height

        # Union
        gt_area = (x2_gt - x1_gt + 1) * (y2_gt - y1_gt + 1)
        det_area = (x2_det - x1_det + 1) * (y2_det - y1_det + 1)
        union = gt_area + det_area - intersection

        iou = intersection / union if union != 0 else 0
        iou_scores.append(iou)

        # Check if IoU >= 0.5 for MAP@0.50
        if iou >= 0.5:
            tp += 1

        # Display IoU on the image
        cv2.putText(frame_gt, f'IoU: {iou:.2f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    else:
        print(f"No circles (rectangles) detected in image {image_file}.")
        # No detection means IoU = 0, no TP increment.
        iou_scores.append(0)

    # Display the frame with detections and ground truth
    cv2.imshow('Golf Ball Detection and Ground Truth', frame_gt)

    # Wait for key press to proceed to next image
    key = cv2.waitKey(10000)
    if key == ord('q'):
        break

# Release resources
cv2.destroyAllWindows()

# Compute average IoU
average_iou = sum(iou_scores) / len(iou_scores) if iou_scores else 0
print(f"Average IoU over {len(iou_scores)} images: {average_iou:.2f}")

# Compute mAP@0.50 (in this simplified scenario it is basically TP/total_images)
map_50 = tp / total_images if total_images > 0 else 0
print(f"mAP@0.50: {map_50:.2f}")
