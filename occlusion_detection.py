import numpy as np

def detect_occlusion(iou_vector, threshold=0.3):
    """Detects occlusion if multiple objects have IoU above a threshold."""
    if not iou_vector or max(iou_vector) == 0:
        return False  # No valid detections

    max_iou = max(iou_vector)  # Main object
    occluding_objects = [iou for iou in iou_vector if iou > threshold and abs(iou - max_iou) > 0.05]
    
    return len(occluding_objects) > 1  # More than 1 object is overlapping
