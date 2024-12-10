import cv2
import numpy as np
import pandas as pd


def parse_yolo_bbox(bbox_line, frame_width, frame_height):
    """
    Convert YOLO format (class x_center y_center width height) to pixel coordinates
    """
    parts = list(map(float, bbox_line.strip().split()))
    x_center, y_center, width, height = parts[1:]
    
    x1 = int((x_center - width/2) * frame_width)
    y1 = int((y_center - height/2) * frame_height)
    x2 = int((x_center + width/2) * frame_width)
    y2 = int((y_center + height/2) * frame_height)
    
    return (x1, y1, x2, y2)

def process_frame_pair(frame1, frame2, roi, time_between_frames):
    """
    Process a pair of frames to compute rotation axis, angular velocity, and RMSD within the ROI.
    """
    # Convert frames to grayscale
    frame1_gray_full = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame2_gray_full = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Apply ROI mask
    x1, y1, x2, y2 = roi
    frame1_gray = frame1_gray_full[y1:y2, x1:x2]
    frame2_gray = frame2_gray_full[y1:y2, x1:x2]

    # Detect the circle within the bounding box using Hough Transform
    circles = cv2.HoughCircles(
        frame1_gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=30,
        param1=50,
        param2=30,
        minRadius=20,
        maxRadius=100
    )

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        circle_x, circle_y, radius = circles[0]
    else:
        return None, None, None, None

    # Create a circular mask
    mask = np.zeros_like(frame1_gray)
    cv2.circle(mask, (circle_x, circle_y), radius, 255, -1)

    # Apply mask to grayscale images
    frame1_gray = cv2.bitwise_and(frame1_gray, frame1_gray, mask=mask)
    frame2_gray = cv2.bitwise_and(frame2_gray, frame2_gray, mask=mask)

    # Compute optical flow within the masked ROI
    prev_pts = cv2.goodFeaturesToTrack(frame1_gray, maxCorners=150, qualityLevel=0.75, minDistance=45, mask=mask)
    if prev_pts is None:
        return None, None, None, None

    # Adjust points to the original frame coordinates
    prev_pts += np.array([[x1, y1]], dtype=np.float32)

    next_pts, status, err = cv2.calcOpticalFlowPyrLK(frame1_gray_full, frame2_gray_full, prev_pts, None)

    # Filter good points
    status = status.reshape(-1)
    prev_pts = prev_pts.reshape(-1, 2)
    next_pts = next_pts.reshape(-1, 2)
    good_prev_pts = prev_pts[status == 1]
    good_next_pts = next_pts[status == 1]

    if len(good_prev_pts) < 3:
        return None, None, None, None

    # Calculate radius and adjust to the detected circle
    img_center = np.array([x1 + circle_x, y1 + circle_y])
    radius = radius

    # Compute x, y coordinates relative to image center
    x = good_prev_pts[:, 0] - img_center[0]
    y = -(good_prev_pts[:, 1] - img_center[1])
    x_next = good_next_pts[:, 0] - img_center[0]
    y_next = -(good_next_pts[:, 1] - img_center[1])

    # Compute z coordinates
    z_squared = radius**2 - x**2 - y**2
    z_squared_next = radius**2 - x_next**2 - y_next**2

    # Filter valid points where z is real
    valid_idx = np.logical_and(z_squared >= 0, z_squared_next >= 0)
    if np.count_nonzero(valid_idx) < 3:
        return None, None, None, None

    x = x[valid_idx]
    y = y[valid_idx]
    z = np.sqrt(z_squared[valid_idx])
    points_3d_prev = np.stack([x, y, z], axis=1)

    x_next = x_next[valid_idx]
    y_next = y_next[valid_idx]
    z_next = np.sqrt(z_squared_next[valid_idx])
    points_3d_next = np.stack([x_next, y_next, z_next], axis=1)

    # Calculate rotation using Kabsch algorithm
    centroid_prev = np.mean(points_3d_prev, axis=0)
    centroid_next = np.mean(points_3d_next, axis=0)
    P = points_3d_prev - centroid_prev
    Q = points_3d_next - centroid_next

    H = P.T @ Q
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Handle reflection case
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    # Calculate rotation angle and axis
    angle = np.arccos((np.trace(R) - 1) / 2)
    
    #R /= R[2, 2]

    #angle = np.arccos(R[0, 0])
    #angle = np.arctan2(R[1, 0], R[0, 0])
    if np.isnan(angle) or angle == 0:
        return None, None, None, None

    axis = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1]
    ]) / (2 * np.sin(angle))

    axis = axis / np.linalg.norm(axis)
    angular_velocity = angle / time_between_frames

    # Calculate RMSD
    points_3d_prev_transformed = (points_3d_prev @ R.T) + (centroid_next - centroid_prev @ R.T)
    residuals = points_3d_prev_transformed - points_3d_next
    squared_distances = np.sum(residuals**2, axis=1)
    rmsd = np.sqrt(np.mean(squared_distances))

    # Prepare optical flow visualization
    flow_image = cv2.cvtColor(frame1_gray_full.copy(), cv2.COLOR_GRAY2BGR)
    for (pt1, pt2) in zip(good_prev_pts, good_next_pts):
        # Ensure points are integer tuples
        a = tuple(map(int, pt1))
        b = tuple(map(int, pt2))
        cv2.arrowedLine(flow_image, a, b, (0, 255, 0), 5, tipLength=0.6)

    return axis, angular_velocity, rmsd, flow_image

def analyze_video_frames(video_path, bbox_txt_path, start_frame=0, visualize=True):
    """
    Analyze sequential frames in a video with YOLO-format bounding box and optional visualization
    """
    # Load video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return

    # Read bounding box file
    with open(bbox_txt_path, 'r') as f:
        bbox_line = f.readline()

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 240  # Set your actual FPS here
    time_between_frames = 1 / fps

    # Convert YOLO bbox to pixel coordinates
    roi = parse_yolo_bbox(bbox_line, frame_width, frame_height)

    frame_idx = 0
    prev_frame = None
    results = []

    out_video = None
    if visualize:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter('optical_flow_output.mp4', fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if prev_frame is not None:
            # Process frame pair
            result = process_frame_pair(
                prev_frame,
                frame,
                roi,
                time_between_frames
            )

            if result[0] is not None:
                axis, angular_velocity, rmsd, flow_image = result
                results.append({
                    'frame_start': frame_idx - 1,
                    'frame_end': frame_idx,
                    'rotation_axis_x': axis[0],
                    'rotation_axis_y': axis[1],
                    'rotation_axis_z': axis[2],
                    'angular_velocity': angular_velocity,
                    'rmsd': rmsd
                })

                # Visualization
                if visualize:
                    # Draw bounding box
                    cv2.rectangle(flow_image, (roi[0], roi[1]), (roi[2], roi[3]), (0, 0, 255), 6)
                    
                    # Display the video frame with optical flow
                    cv2.imshow('Optical Flow Visualization', flow_image)

                    # Write frame to output video
                    if out_video is not None:
                        out_video.write(flow_image)

                    # Wait for a key press (Press 'q' to quit)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        prev_frame = frame.copy()
        frame_idx += 1

    cap.release()
    if out_video is not None:
        out_video.release()
    cv2.destroyAllWindows()  # Close all OpenCV windows

    # Check if any results were collected
    if not results:
        print("No valid frame pairs were processed.")
        return

    # Create DataFrame and save to CSV
    df = pd.DataFrame(results)
    df.to_csv('/Users/christianwarburg/Desktop/Fagproject/miguel_exp/logoball/zerotilt/10rpm_rotation.csv', index=False)
    print("Analysis complete. Results saved to rotation_analysis.csv")
    return df

# Example usage
video_path = "/Users/christianwarburg/Desktop/Fagproject/miguel_exp/logoball/zerotilt/trimmed100rpm.mov"
bbox_txt_path = "/Users/christianwarburg/Desktop/Fagproject/miguel_exp/logoball/25tilt/labels/100rpm.txt"
results_df = analyze_video_frames(video_path, bbox_txt_path)



