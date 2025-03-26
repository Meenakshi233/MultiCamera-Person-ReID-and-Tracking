import requests

SERVER_URL = "http://localhost:5000/track"  # Replace with actual URL

def inter_camera_tracking(target_features, cam_id):
    """Handles tracking across multiple cameras with error handling"""
    try:
        response = requests.post(SERVER_URL, json={
            "camera_id": cam_id,
            "features": target_features.tolist()
        })
        
        # Ensure response is valid JSON
        response.raise_for_status()  # Raises an error for HTTP codes 4xx/5xx
        return response.json()  # Return parsed JSON response

    except requests.exceptions.RequestException as e:
        print(f"Error contacting server: {e}")
        return {"error": "Failed to contact server"}
