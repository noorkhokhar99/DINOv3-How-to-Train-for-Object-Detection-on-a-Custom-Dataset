import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import json
from tqdm import tqdm
import os
import glob
import pyresearch

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

# Image transformation pipeline
transform_image = T.Compose([
    T.ToTensor(),
    T.Resize(256),
    T.CenterCrop(224),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def load_image(img_path: str) -> torch.Tensor:
    """
    Load an image and return a tensor for DINOv3 input.
    """
    try:
        img = Image.open(img_path).convert("RGB")
        transformed_img = transform_image(img)[:3].unsqueeze(0)
        return transformed_img
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")
        return None

def compute_embeddings(images_base_path: str, split: str) -> dict:
    """
    Compute DINOv3 embeddings for images in the specified split's subfolders.
    Args:
        images_base_path: Base path to the dataset (e.g., containing train, valid, test).
        split: Name of the dataset split (e.g., 'train', 'valid', 'test').
    Returns:
        Dictionary mapping image paths to embeddings and their class.
    """
    all_embeddings = {}
    split_path = os.path.join(images_base_path, split)
    if not os.path.exists(split_path):
        print(f"Folder {split_path} does not exist, skipping...")
        return all_embeddings

    # Find all subfolders (classes)
    class_folders = [f for f in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, f))]
    if not class_folders:
        print(f"No subfolders found in {split_path}")
        return all_embeddings

    # Support multiple image formats
    image_extensions = ["*.jpg", "*.jpeg", "*.png"]
    for class_name in class_folders:
        class_path = os.path.join(split_path, class_name)
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(class_path, ext)))
        
        if not image_files:
            print(f"No images found in {class_path}")
            continue

        with torch.no_grad():
            for file in tqdm(image_files, desc=f"Computing embeddings for {split}/{class_name}"):
                # Compute embedding
                img_tensor = load_image(file)
                if img_tensor is None:
                    continue
                embeddings = dinov3_vits16(img_tensor.to(device))
                embeddings = np.array(embeddings[0].cpu().numpy()).reshape(1, -1).tolist()
                all_embeddings[file] = {"embeddings": embeddings, "class": class_name}

    # Save embeddings to JSON, avoid overwriting
    output_file = f"{split}_dinov3_vits16_embeddings.json"
    if os.path.exists(output_file):
        print(f"Warning: {output_file} already exists. Saving to {split}_dinov3_vits16_embeddings_new.json instead.")
        output_file = f"{split}_dinov3_vits16_embeddings_new.json"
    
    with open(output_file, "w") as f:
        json.dump(all_embeddings, f)
    print(f"Embeddings for {split} saved to {output_file}")
    return all_embeddings

if __name__ == "__main__":
    # Base path to the dataset
    base_path = "C:/Users/ASUS/Desktop/pyresearch videos codes/Sign_Language_Detectiom"  # Matches your command path

    # List of splits to process
    splits = ["train", "valid", "test"]

    # Compute embeddings for each split
    all_embeddings = {}
    for split in splits:
        all_embeddings[split] = compute_embeddings(base_path, split)