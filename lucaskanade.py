import cv2
import numpy as np
import pandas as pd

# Load bounding box data from CSV
bounding_boxes = pd.read_csv('/Users/christianwarburg/Desktop/Fagproject/Fagproject2024/predictions.csv')  # Update with your path

# Parameters for ShiTomasi corner detection (goodFeaturesToTrack)
feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Capture video
cap = cv2.VideoCapture('/Users/christianwarburg/Desktop/Fagproject/tuxenkort.mp4')

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object (optional, if you want to save the output)
save_output = False  # Set to True if you want to save the video with optical flow
if save_output:
    out = cv2.VideoWriter('output_with_optical_flow.mp4',
                          cv2.VideoWriter_fourcc(*'mp4v'),
                          fps, (frame_width, frame_height))

# Read the first frame
ret, old_frame = cap.read()
if not ret:
    print("Error: Cannot read video file.")
    cap.release()
    exit()

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# Collect optical flow vectors for plotting (optional)
optical_flow_vectors = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Get the current frame number
    frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
    frame_name = f"tuxen_frames/frame_{frame_idx}.jpg"
    
    # Filter bounding boxes for the current frame
    bbox = bounding_boxes[bounding_boxes['image_path'] == frame_name]
    
    if not bbox.empty:
        x1, y1, x2, y2 = int(bbox.iloc[0]['x1']), int(bbox.iloc[0]['y1']), int(bbox.iloc[0]['x2']), int(bbox.iloc[0]['y2'])
        
        # Draw the bounding box on the frame
        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)
        
        # Create a mask for the bounding box region
        bbox_mask = np.zeros_like(old_gray)
        bbox_mask[y1:y2, x1:x2] = 255
        
        # Detect corners within the bounding box
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=bbox_mask, **feature_params)

        if p0 is not None:
            # Calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
            
            # Select good points
            if p1 is not None and st is not None:
                good_new = p1[st == 1]
                good_old = p0[st == 1]
                
                # Draw the tracks
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    # Draw optical flow vectors
                    frame = cv2.arrowedLine(frame, (int(c), int(d)), (int(a), int(b)),
                                            color=(0, 255, 0), thickness=2, tipLength=0.5)
                    # Draw circles at feature points
                    frame = cv2.circle(frame, (int(a), int(b)), 3, color=(0, 0, 255), thickness=-1)
                    
    else:
        # If no bounding box for this frame, you can choose to skip or process the whole frame
        pass  # For now, we do nothing

    # Display the frame
    cv2.imshow('Optical Flow with Bounding Box', frame)
    
    if save_output:
        out.write(frame)
    
    # Press 'q' to exit the video early
    if cv2.waitKey(1010) & 0xFF == ord('q'):
        break
    
    # Update for next iteration
    old_gray = frame_gray.copy()
    if 'good_new' in locals():
        p0 = good_new.reshape(-1, 1, 2)
    else:
        p0 = None

cap.release()
if save_output:
    out.release()
cv2.destroyAllWindows()

a= 2
