import cv2
import numpy as np
from feature_extraction import extract_features
from occlusion_detection import detect_occlusion

tracker = cv2.TrackerKCF_create()  # Change to another tracker if needed

def calculate_iou(features1, features2):
    """Calculates IoU between two feature vectors"""
    intersection = np.minimum(features1, features2).sum()
    union = np.maximum(features1, features2).sum()
    return intersection / union if union > 0 else 0

def intra_camera_tracking(frame, bboxes, cam_id):
    """Handles tracking within a single camera"""
    global tracker
    if len(bboxes) > 0:
        features = extract_features(frame, bboxes)
        target_features = extract_features(frame, [bboxes[0]])
        
        iou_scores = [calculate_iou(target_features, extract_features(frame, [bbox])) for bbox in bboxes]

        x1, y1, x2, y2 = bboxes[0]
        bbox = (x1, y1, x2 - x1, y2 - y1)

        if max(iou_scores) < 0.3 or detect_occlusion(iou_scores):
            tracker = cv2.TrackerKCF_create()
            tracker.init(frame, bbox)  # Initialize tracker
        else:
            success, new_box = tracker.update(frame)
            if success:
                x, y, w, h = new_box
                cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
