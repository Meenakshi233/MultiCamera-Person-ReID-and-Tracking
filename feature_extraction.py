import numpy as np
import cv2
import torch
from torchvision import models, transforms

# Load a pre-trained CNN model (e.g., ResNet50)
model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove classification layer
model.eval()

# Image preprocessing function
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),  # Resize to match CNN input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_features(frame, bboxes):
    """Extracts deep learning-based features from bounding boxes."""
    if not isinstance(frame, np.ndarray):
        raise ValueError("Frame must be a NumPy array")
    
    features = []
    for bbox in bboxes:
        x1, y1, x2, y2 = map(int, bbox)
        
        # Ensure bounding box is within frame dimensions
        h, w, _ = frame.shape
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        
        if x1 >= x2 or y1 >= y2:
            continue  # Skip invalid bounding boxes
        
        cropped = frame[y1:y2, x1:x2]  # Extract the region
        cropped = transform(cropped).unsqueeze(0)  # Preprocess
        
        with torch.no_grad():
            feature = model(cropped).squeeze().numpy()  # Extract feature
        
        features.append(feature)
    
    return np.array(features) if features else np.array([])  # Handle empty case
