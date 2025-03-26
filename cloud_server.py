from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

# Dictionary to store extracted features {person_id: feature_vector}
feature_database = {}

def cosine_similarity(vec1, vec2):
    """Calculate Cosine Similarity between two vectors"""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

@app.route("/track", methods=["POST"])
def track_person():
    data = request.json
    camera_id = data["camera_id"]
    new_feature = np.array(data["features"])

    best_match = None
    max_similarity = 0.0
    threshold = 0.8  # Adjust based on testing

    for person_id, stored_feature in feature_database.items():
        similarity = cosine_similarity(new_feature, stored_feature)
        if similarity > max_similarity:
            max_similarity = similarity
            best_match = person_id

    if max_similarity > threshold:
        result = {"camera_id": camera_id, "status": "tracking", "person_id": best_match}
    else:
        person_id = len(feature_database) + 1  # Assign new ID
        feature_database[person_id] = new_feature
        result = {"camera_id": camera_id, "status": "new_person_detected", "person_id": person_id}

    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
