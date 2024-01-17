import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def compute_ssim(rendering_gt, rendering_reconstructed, data_range=1.0):
    # The images passed to the function should already be normalized
    ssim_value = ssim(rendering_gt, rendering_reconstructed, data_range=data_range)
    
    return ssim_value

def compute_psnr(gt_image, rendered_image):
    # The images passed to the function should already be normalized
    gt_image = gt_image.astype(np.float32)
    rendered_image = rendered_image.astype(np.float32)

    '''
    mse = np.mean((gt_image - rendered_image) ** 2)
    psnr = 20 * np.log10(1.0 / np.sqrt(mse))
    '''

    psnr_value = psnr(gt_image, rendered_image, data_range=rendered_image.max() - rendered_image.min())

    return psnr_value