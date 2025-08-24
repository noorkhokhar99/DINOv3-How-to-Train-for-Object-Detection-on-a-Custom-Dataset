import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import json
import os
import cv2  # For camera access and display
from scipy.spatial.distance import cdist  # For distance computation
from collections import Counter  # For majority voting in KNN

# Path to the local DINOv3 repository
REPO_DIR = "C:/Users/ASUS/Desktop/pyresearch videos codes/Sign_Language_Detectiom/dinov3"  # Path to cloned repository
# Path to the downloaded weights
WEIGHTS_PATH = "C:/Users/ASUS/Desktop/pyresearch videos codes/Sign_Language_Detectiom/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"

# Initialize DINOv3 model
try:
    dinov3_vits16 = torch.hub.load(REPO_DIR, 'dinov3_vits16', source='local', weights=WEIGHTS_PATH)
except Exception as e:
    print(f"Error loading DINOv3 model: {e}")
    print("Ensure REPO_DIR points to the cloned DINOv3 repository and WEIGHTS_PATH is correct.")
    exit(1)

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
dinov3_vits16.to(device)
dinov3_vits16.eval()  # Set model to evaluation mode

# Image transformation pipeline (adjusted for DINOv3)
transform_image = T.Compose([
    T.ToTensor(),
    T.Resize(256),
    T.CenterCrop(224),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load train embeddings from JSON
def load_train_embeddings(json_path: str):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"{json_path} not found. Compute embeddings first.")
    with open(json_path, 'r') as f:
        data = json.load(f)
    embeddings = []
    labels = []
    for info in data.values():
        embeddings.append(info['embeddings'][0])  # Extract the inner list
        labels.append(info['class'])
    return np.array(embeddings), np.array(labels)

# Function to compute embedding for a new image (from frame)
def compute_embedding(frame: np.ndarray) -> np.ndarray:
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Convert OpenCV frame to PIL
    transformed_img = transform_image(img)[:3].unsqueeze(0).to(device)
    with torch.no_grad():
        emb = dinov3_vits16(transformed_img)
    return emb.cpu().numpy().reshape(1, -1)  # Reshape to (1, embedding_dim)

# KNN prediction function
def predict_class(new_embedding: np.ndarray, train_embeddings: np.ndarray, train_labels: np.ndarray, k: int = 5) -> str:
    dists = cdist(new_embedding, train_embeddings, metric='cosine')  # Cosine distance (lower is better)
    idx = np.argsort(dists[0])[:k]  # Get indices of k nearest neighbors
    nearest_classes = train_labels[idx]
    pred_class = Counter(nearest_classes).most_common(1)[0][0]  # Majority vote
    return pred_class

if __name__ == "__main__":
    # Path to your train embeddings JSON
    train_json_path = "test_dinov3_vits16_embeddings.json"
    train_embeddings, train_labels = load_train_embeddings(train_json_path)
    print(f"Loaded {len(train_embeddings)} train embeddings.")

    # Camera settings
    camera_index = 1  # Try 0, 1, or 2 if needed
    use_file_fallback = False  # Set to True and provide video_path below for testing without camera
    video_path = "C:/Users/ASUS/Downloads/Sign_Language_Detectiom/test_video.mp4"  # For fallback testing

    if use_file_fallback:
        cap = cv2.VideoCapture(video_path)
    else:
        # Use CAP_DSHOW backend for better Windows compatibility
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        # Set lower resolution to avoid memory/format issues
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera (index {camera_index}). Check connections, permissions, or try a different index/backend.")

    frame_skip = 5  # Process every 5th frame for efficiency (adjust as needed)
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or camera feed interrupted.")
            break
        frame_count += 1
        if frame_count % frame_skip == 0:
            # Compute embedding and predict class
            new_emb = compute_embedding(frame)
            predicted_class = predict_class(new_emb, train_embeddings, train_labels, k=5)
            
            # Display class on frame
            cv2.putText(frame, f"Class: {predicted_class}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show the frame
        cv2.imshow('Sign Language Detection', frame)
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()