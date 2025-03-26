import cv2
import torch
import numpy as np
from intra_tracking import intra_camera_tracking
from inter_tracking import inter_camera_tracking
from yolov5.models.common import DetectMultiBackend

# Load YOLOv5 model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DetectMultiBackend("yolov5s.pt", device=device)  # Ensure yolov5s.pt is in the same directory
model.eval()

# Read RTSP URLs from file
with open("camera_sources.txt", "r") as file:
    cameras = [line.strip() for line in file]

camera_streams = [cv2.VideoCapture(url) for url in cameras]

def process_frame(frame, cam_id):
    """Process a single frame for target detection and tracking"""
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img = torch.from_numpy(img).permute(2, 0, 1).float().to(device) / 255.0  # Normalize
    img = img.unsqueeze(0)  # Add batch dimension
    
    with torch.no_grad():
        results = model(img)

    # Extract bounding boxes
    bboxes = []
    for det in results.xyxy[0].cpu().numpy():  # Extract detections
        x1, y1, x2, y2, conf, cls = det
        if conf > 0.3:  # Confidence threshold
            bboxes.append([x1, y1, x2, y2])

    if bboxes:
        intra_camera_tracking(frame, bboxes, cam_id)

        # Extract features for inter-camera tracking (Modify based on feature extraction)
        target_features = np.array(bboxes[0])  # Placeholder: Replace with real features
        inter_camera_tracking(target_features, cam_id)

while True:
    for i, cam in enumerate(camera_streams):
        ret, frame = cam.read()
        if not ret:
            print(f"Camera {i} disconnected, retrying...")
            cam.release()
            camera_streams[i] = cv2.VideoCapture(cameras[i])
            continue

        process_frame(frame, i)
        cv2.imshow(f"Camera {i}", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release cameras and close windows
for cam in camera_streams:
    cam.release()
cv2.destroyAllWindows()
