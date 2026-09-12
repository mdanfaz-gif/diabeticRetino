"""
Image Quality Assessment (IQA) & Recapture Guidance Engine
Block 3 of the SIH26038 Architecture + Architectural Enhancement 1.

Features:
- Blur and Sharpness Detection (Variance of Laplacian)
- Illumination and Exposure Check (Under/Over-exposure ratios)
- Field of View (FOV) Retinal Coverage Validation
- Anatomical Landmark Detection (Optic Disc & Fovea/Macula)
- Composite Quality Scoring (0 - 100)
- Actionable Recapture Guidance for Rural Health Workers (ASHA/ANM)
"""

import cv2
import numpy as np

def compute_sharpness_score(img_rgb, fov_mask):
    """
    Computes sharpness using the Variance of the Laplacian within the retinal FOV.
    Higher values indicate sharp vessel boundaries; lower values indicate motion blur.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    
    # Calculate variance strictly within the retinal FOV
    fov_indices = fov_mask > 0
    if np.sum(fov_indices) == 0:
        return 0.0
        
    masked_lap = laplacian[fov_indices]
    var_lap = float(np.var(masked_lap))
    
    # Normalize variance to a 0 - 100 scale (typically 0-500 in fundus photography)
    sharpness_score = min(100.0, (var_lap / 250.0) * 100.0)
    return sharpness_score, var_lap

def compute_illumination_metrics(img_rgb, fov_mask):
    """
    Evaluates exposure balance, clipping, and dynamic range.
    Detects glare/flash reflection (overexposure) and dark capture (underexposure).
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    fov_pixels = gray[fov_mask > 0]
    
    if len(fov_pixels) == 0:
        return {"mean_intensity": 0, "under_ratio": 1.0, "over_ratio": 1.0, "illumination_score": 0}
        
    mean_val = float(np.mean(fov_pixels))
    under_ratio = float(np.sum(fov_pixels < 35) / len(fov_pixels))
    over_ratio = float(np.sum(fov_pixels > 240) / len(fov_pixels))
    
    # Ideal mean intensity for fundus is typically 80 - 160
    if 80 <= mean_val <= 165 and over_ratio < 0.05 and under_ratio < 0.15:
        illum_score = 95.0
    elif 60 <= mean_val <= 190 and over_ratio < 0.10 and under_ratio < 0.25:
        illum_score = 75.0
    elif under_ratio > 0.40:
        illum_score = max(10.0, 50.0 - under_ratio * 70)
    elif over_ratio > 0.15:
        illum_score = max(10.0, 50.0 - over_ratio * 100)
    else:
        illum_score = 55.0
        
    return {
        "mean_intensity": round(mean_val, 1),
        "under_ratio": round(under_ratio, 3),
        "over_ratio": round(over_ratio, 3),
        "illumination_score": round(illum_score, 1)
    }

def detect_anatomical_landmarks(img_rgb, fov_mask):
    """
    Detects Optic Disc (brightest circular region) and Fovea/Macula (dark avascular zone).
    Validates whether the fundus capture contains standard diagnostic landmarks.
    """
    h, w, _ = img_rgb.shape
    green = img_rgb[:, :, 1]
    
    # 1. Optic Disc Candidate Search: High intensity in Red channel & morphological closing
    red = img_rgb[:, :, 0]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    red_close = cv2.morphologyEx(red, cv2.MORPH_CLOSE, kernel)
    
    # Mask out background
    masked_red = cv2.bitwise_and(red_close, red_close, mask=fov_mask)
    
    # Find position of highest intensity cluster (Optic Disc)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(masked_red)
    od_x, od_y = max_loc
    
    # Approximate disc diameter based on image width (typically ~1/6 to 1/8 of fundus width)
    od_radius = int(w * 0.07)
    
    # 2. Fovea / Macula Detection:
    # Fovea is temporal to the optic disc (either left or right depending on OS/OD eye),
    # approximately 2.5 to 3 disc diameters away, and is the local green-channel minimum.
    # Determine side based on OD position relative to center:
    center_x = w // 2
    if od_x > center_x:
        # OD on right -> Macula is to the left (Right Eye / OD fundus)
        macula_est_x = max(int(w * 0.2), od_x - int(od_radius * 2.8))
    else:
        # OD on left -> Macula is to the right (Left Eye / OS fundus)
        macula_est_x = min(int(w * 0.8), od_x + int(od_radius * 2.8))
        
    macula_est_y = od_y + int(h * 0.02) # Slightly lower or level with OD
    
    # Refine Macula coordinates around local minimum in green channel
    crop_r = int(od_radius * 0.8)
    y1, y2 = max(0, macula_est_y - crop_r), min(h, macula_est_y + crop_r)
    x1, x2 = max(0, macula_est_x - crop_r), min(w, macula_est_x + crop_r)
    
    if y2 > y1 and x2 > x1:
        local_crop = green[y1:y2, x1:x2]
        _, _, min_loc_local, _ = cv2.minMaxLoc(local_crop)
        fovea_x = x1 + min_loc_local[0]
        fovea_y = y1 + min_loc_local[1]
    else:
        fovea_x, fovea_y = macula_est_x, macula_est_y
        
    landmarks_valid = (max_val > 140 and fov_mask[od_y, od_x] > 0 and fov_mask[fovea_y, fovea_x] > 0)
    
    return {
        "optic_disc": (int(od_x), int(od_y), int(od_radius)),
        "fovea": (int(fovea_x), int(fovea_y)),
        "landmarks_valid": bool(landmarks_valid)
    }

def assess_image_quality(img_rgb):
    """
    Comprehensive Image Quality Assessment (Block 3).
    Returns composite score, acceptability flag, and actionable guidance for rural staff.
    """
    h, w, _ = img_rgb.shape
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    
    # 1. FOV Mask & Coverage
    _, mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    fov_pixels = np.sum(mask > 0)
    total_pixels = h * w
    fov_ratio = fov_pixels / total_pixels
    
    # 2. Sharpness & Blur
    sharpness_score, var_lap = compute_sharpness_score(img_rgb, mask)
    
    # 3. Illumination & Exposure
    illum_data = compute_illumination_metrics(img_rgb, mask)
    
    # 4. Anatomical Landmarks
    landmarks = detect_anatomical_landmarks(img_rgb, mask)
    
    # 5. Composite Quality Score (0 - 100)
    # Weights: Sharpness (45%), Illumination (35%), FOV & Landmarks (20%)
    fov_score = 95.0 if (0.35 <= fov_ratio <= 0.85 and landmarks["landmarks_valid"]) else 50.0
    
    composite_score = (
        0.45 * sharpness_score +
        0.35 * illum_data["illumination_score"] +
        0.20 * fov_score
    )
    composite_score = round(float(np.clip(composite_score, 5.0, 99.0)), 1)
    
    # 6. Quality Acceptable Decision (Threshold = 60)
    is_acceptable = composite_score >= 60.0
    
    # 7. Actionable Recapture Guidance
    guidance = []
    if sharpness_score < 45.0:
        guidance.append("Motion blur detected. Ensure the patient's chin and forehead rest firmly against the frame. Hold the camera steady for 2 seconds before capture.")
    if illum_data["under_ratio"] > 0.25 or illum_data["mean_intensity"] < 65:
        guidance.append("Image is underexposed / dark. Increase fundus adapter LED brightness and verify patient pupil dilation (minimum 4mm recommended).")
    if illum_data["over_ratio"] > 0.08:
        guidance.append("Corneal glare / specular reflection detected. Dim the ambient room lighting and slightly tilt the adapter axis away from reflection.")
    if fov_ratio < 0.35 or not landmarks["landmarks_valid"]:
        guidance.append("Retinal field is clipped or off-center. Align the camera preview crosshair directly onto the center of the pupil.")
    if is_acceptable and not guidance:
        guidance.append("Image quality is optimal. Diagnostic clarity meets ETDRS telemedicine standards.")

    return {
        "score": composite_score,
        "is_acceptable": is_acceptable,
        "sharpness": round(sharpness_score, 1),
        "laplacian_var": round(var_lap, 1),
        "illumination": illum_data["illumination_score"],
        "mean_intensity": illum_data["mean_intensity"],
        "overexposed_ratio": illum_data["over_ratio"],
        "underexposed_ratio": illum_data["under_ratio"],
        "fov_coverage_ratio": round(fov_ratio, 3),
        "landmarks": landmarks,
        "guidance": guidance
    }
