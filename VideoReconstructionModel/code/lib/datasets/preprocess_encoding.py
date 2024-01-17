import numpy as np
import torch
import cv2
import os
from pretrained_encoders import EncodingBackbone
from torchvision import transforms
from PIL import Image


encoder = EncodingBackbone()
encoder.eval()

video_folder = "/AGen/data/3DPW_x_AGen/data/test/outdoors_fencing_01"
video_path = os.path.join(video_folder, "image")

print(f"looking in folder: {video_path}")

image_files = sorted([f for f in os.listdir(video_path) if f.endswith(('.jpg', '.png'))])

print(image_files)

feature_encoding_vectors = []

for image_file in image_files:
    # Construct the full path to the image file
    image_path = os.path.join(video_path, image_file)

    # Read the image using OpenCV
    frame = cv2.imread(image_path)

    # Convert the frame to the format expected by the encoder
    # (e.g., convert BGR to RGB and normalize pixel values)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame / 255.0  # Normalize pixel values to [0, 1]

    # Resize the image using torchvision
    resized_image = cv2.resize(frame, (224, 224))
    
    # Convert the frame to a PyTorch tensor
    #frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float()
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    #normalized_frame_tensor = transforms.functional.normalize(frame_tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    input_image = preprocess(resized_image).unsqueeze(0).float()

    with torch.no_grad():
        feature_encoding = encoder(input_image)
    
    feature_encoding = feature_encoding.view(-1)
    feature_encoding_vectors.append(feature_encoding.numpy())

    print(f"Encoding vector shape for {image_file}: {feature_encoding.shape}")
    #print(f"Encoding vector for {image_file}: {feature_encoding}")

output_file = "feature_encoding_vectors.npy"
output_file = os.path.join(video_folder, output_file)
np.save(output_file, feature_encoding_vectors)