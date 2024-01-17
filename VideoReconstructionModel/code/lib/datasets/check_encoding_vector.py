import numpy as np
import torch
import cv2
import os
from pretrained_encoders import EncodingBackbone
from torchvision import transforms


encoder = EncodingBackbone()

video_folder = "/AGen/data/3DPW_x_AGen/data/train/courtyard_backpack_00"
video_path = os.path.join(video_folder, "image")

image_files = sorted([f for f in os.listdir(video_path) if f.endswith(('.jpg', '.png'))])

feature_encoding_vectors = []

for i, image_file in enumerate(image_files):
    # Construct the full path to the image file
    image_path = os.path.join(video_path, image_file)

    # Read the image using OpenCV
    frame = cv2.imread(image_path)

    # Convert the frame to the format expected by the encoder
    # (e.g., convert BGR to RGB and normalize pixel values)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame / 255.0  # Normalize pixel values to [0, 1]
    
    # Convert the frame to a PyTorch tensor
    frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float()
    normalized_frame_tensor = transforms.functional.normalize(frame_tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    with torch.no_grad():
        feature_encoding = encoder(normalized_frame_tensor)
    
    feature_encoding_vectors.append(feature_encoding.numpy())

    npy_path = os.path.join(video_folder, "feature_encoding_vectors.npy")

    print("npy_path:", npy_path)

    # Load the encoding vector from the .npy file
    encoding_vector = np.load(npy_path)

    # Compare the encoding vector from the .npy file to the encoding vector from the encoder
    print("Computed encoding vector:", feature_encoding.numpy())
    print(f"Encoding vector from .npy file: {encoding_vector[i]}")
    print("They are equal:", np.array_equal(feature_encoding, encoding_vector[i]))

#output_file = "feature_encoding_vectors.npy"
#output_file = os.path.join(video_folder, output_file)
#np.save(output_file, feature_encoding_vectors)