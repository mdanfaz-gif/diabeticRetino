"""
Image Preprocessing Pipeline (MATLAB / OpenCV Equivalent)
Block 2 of the SIH26038 Architecture.

Functions:
- Circular Retinal Field (FOV) Segmentation
- Green Channel Extraction
- Contrast Limited Adaptive Histogram Equalization (CLAHE)
- Graham's Illumination and Color Normalization
- Bilateral Noise Reduction
"""

import cv2
import numpy as np

def get_retinal_fov_mask(img_rgb, threshold=15):
    """
    Extract the circular retinal Field of View (FOV) mask.
    Filters out the dark camera background outside the eyeball.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    # Fill any holes and smooth edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Find largest contour which is the retinal disk
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_mask = np.zeros_like(mask)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(clean_mask, [largest_contour], -1, 255, -1)
    else:
        clean_mask = mask
        
    return clean_mask

def extract_green_channel(img_rgb):
    """
    Retinal blood vessels and microaneurysms absorb green light strongly,
    providing the highest contrast against the red/orange choroid background.
    """
    return img_rgb[:, :, 1]

def apply_clahe(img_gray_or_rgb, clip_limit=2.5, tile_grid_size=(8, 8)):
    """
    Apply Contrast Limited Adaptive Histogram Equalization.
    If RGB, applies to the Luminance (L) channel in LAB color space to preserve natural hue.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    if len(img_gray_or_rgb.shape) == 2:
        return clahe.apply(img_gray_or_rgb)
    else:
        lab = cv2.cvtColor(img_gray_or_rgb, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

def normalize_illumination_graham(img_rgb, sigma=30):
    """
    Graham's method for fundus illumination normalization:
    Subtracts a low-pass Gaussian blurred version (local average illumination)
    and adds a constant baseline, compensating for uneven peripheral vignetting.
    """
    blurred = cv2.GaussianBlur(img_rgb, (0, 0), sigma)
    # 4 * img - 4 * blurred + 128
    normalized = cv2.addWeighted(img_rgb, 4.0, blurred, -4.0, 128)
    return normalized

def reduce_noise_bilateral(img_rgb, d=9, sigma_color=75, sigma_space=75):
    """
    Bilateral filter smooths noise while keeping crisp lesion and vessel boundaries.
    """
    return cv2.bilateralFilter(img_rgb, d, sigma_color, sigma_space)

def preprocess_fundus_pipeline(img_rgb):
    """
    Complete Block 2 Preprocessing Pipeline.
    Returns a dict containing intermediate representations and the final enhanced image.
    """
    # 1. FOV Mask
    fov_mask = get_retinal_fov_mask(img_rgb)
    
    # 2. Green Channel
    green_ch = extract_green_channel(img_rgb)
    
    # 3. CLAHE on Green Channel
    clahe_green = apply_clahe(green_ch, clip_limit=3.0)
    
    # 4. Illumination Normalization
    graham_norm = normalize_illumination_graham(img_rgb)
    
    # 5. Noise Reduction
    denoised = reduce_noise_bilateral(img_rgb)
    
    # 6. Preprocessed Composite (LAB CLAHE + Graham Blend within FOV)
    enhanced_lab = apply_clahe(img_rgb, clip_limit=2.5)
    blended = cv2.addWeighted(enhanced_lab, 0.7, graham_norm, 0.3, 0)
    
    # Mask out background to pure black
    preprocessed_final = cv2.bitwise_and(blended, blended, mask=fov_mask)
    
    return {
        "original": img_rgb,
        "fov_mask": fov_mask,
        "green_channel": green_ch,
        "clahe_green": clahe_green,
        "graham_norm": graham_norm,
        "denoised": denoised,
        "preprocessed_final": preprocessed_final
    }
