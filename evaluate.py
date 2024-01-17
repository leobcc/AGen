import torch
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import glob
import argparse
import cv2
import numpy as np
import argparse
from tripy import shoelace_formula
from pyntcloud import PyntCloud
from pyntcloud.distance import euclidean_distances

def evaluate_rendering(args):
    ssim_values = []
    psnr_values = []
    test_rendering_paths = sorted(glob.glob(f'{args.path}/*.png'))
    for test_rendering_path in test_rendering_paths:
        combined_image = cv2.imread(test_rendering_path)

        height, width = combined_image.shape[:2]

        input_image = combined_image[:height // 2, :]
        rendered_image = combined_image[height // 2:, :]

        # Calculate SSIM and PSNR
        ssim_value = ssim(input_image, rendered_image, multichannel=True)
        psnr_value = psnr(input_image, rendered_image)

        ssim_values.append(ssim_value)
        psnr_values.append(psnr_value)

    # Log or print the SSIM and PSNR values
    ssim_average = sum(ssim_values) / len(ssim_values)
    psnr_average = sum(psnr_values) / len(psnr_values)

    print(f'average_SSIM: {ssim_average:.4f}')
    print(f'average_PSNR: {psnr_average:.4f}')

    return

def calculate_iou(mesh1, mesh2):
    # Compute intersection area using Shoelace formula
    intersection_area = shoelace_formula(mesh1.intersection(mesh2))

    # Compute union area using Shoelace formula
    union_area = shoelace_formula(mesh1) + shoelace_formula(mesh2) - intersection_area

    # Calculate Intersection over Union (IoU)
    iou = intersection_area / union_area
    return iou

def calculate_chamfer_distance(mesh1, mesh2):
    # Compute Chamfer distance using Euclidean distances
    distances1 = euclidean_distances(mesh1.vertices, mesh2.vertices).min(axis=1)
    distances2 = euclidean_distances(mesh2.vertices, mesh1.vertices).min(axis=1)
    chamfer_distance = (distances1.mean() + distances2.mean()) / 2
    return chamfer_distance

def calculate_normal_consistency(mesh1, mesh2):
    # Compute normal consistency as the cosine similarity of normals
    cos_similarity = np.abs(np.sum(mesh1.normals * mesh2.normals, axis=1))
    normal_consistency = cos_similarity.mean()
    return normal_consistency

def evaluate_3D(args):
    # Load 3D meshes (replace these paths with your mesh files)
    mesh1 = PyntCloud.from_file('path_to_mesh1.ply')
    mesh2 = PyntCloud.from_file('path_to_mesh2.ply')

    # Calculate IoU
    iou = calculate_iou(mesh1, mesh2)
    print(f'IoU: {iou:.4f}')

    # Calculate Chamfer distance
    chamfer_distance = calculate_chamfer_distance(mesh1, mesh2)
    print(f'Chamfer Distance: {chamfer_distance:.4f}')

    # Calculate normal consistency
    normal_consistency = calculate_normal_consistency(mesh1, mesh2)
    print(f'Normal Consistency: {normal_consistency:.4f}')

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='3D Visualization')
    # evaluation over the rendering or the 3D reconstruction
    parser.add_argument('--mode', type=str, help='mode: rendering or 3D')
    # rendering file path or meshes file path
    parser.add_argument('--path', type=str, help='path to the file')
    args = parser.parse_args()
    if args.mode == 'rendering':
        evaluate_rendering(args)
    elif args.mode == '3D':
        evaluate_3D(args)