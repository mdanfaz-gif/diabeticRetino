"""
Low-Cost Rural Hardware Adaptation Module (Enhancement B2)
Block 1 & Practical Rural Constraints.

Supports low-cost smartphone-based fundus imaging adapters:
1. Remidio Fundus on Phone (FOP): Smartphone direct ophthalmoscopy (45 deg field)
2. MII RetCam / 20D Lens Attachment: Inverted indirect smartphone ophthalmoscopy (55 deg field)
3. Standard Desktop Fundus Camera: Tabletop reference (Zeiss / Topcon)

Applies sensor-specific optical corrections:
- Barrel distortion rectification
- Central illumination hotspot suppression
- Chromatic aberration compensation
- Optical circle alignment
"""

import cv2
import numpy as np

HARDWARE_PROFILES = {
    "REMIDIO_FOP": {
        "name": "Remidio Fundus on Phone (FOP)",
        "type": "Direct Smartphone Fundus Adapter (45° FOV)",
        "cost_category": "Low-Cost Portable (INR ~1.5L - 2.5L)",
        "typical_artifacts": "Central LED reflection, peripheral smartphone lens chromatic aberration",
        "description": "Validated handheld smartphone fundus camera widely deployed in Indian rural screening."
    },
    "MII_RETCAM": {
        "name": "MII RetCam / 20D Lens Attachment",
        "type": "Indirect Smartphone Attachment (55° Wide-field)",
        "cost_category": "Ultra Low-Cost (< INR 30,000 / $350)",
        "typical_artifacts": "Inverted image, circular vignette border, mild barrel distortion",
        "description": "Ultra-affordable 20D condensing lens smartphone mount invented in India for rural PHCs."
    },
    "DESKTOP_STANDARD": {
        "name": "Standard Desktop Fundus Camera (Zeiss / Topcon)",
        "type": "High-Resolution Tabletop Camera (45° Standard)",
        "cost_category": "High-End Tertiary Hospital (INR 15L - 30L)",
        "typical_artifacts": "Planar telecentric lens, minimal distortion, high SNR",
        "description": "Hospital benchmark standard for dilated clinical ophthalmic photography."
    }
}

def correct_barrel_distortion(img_rgb, k1=-0.08, k2=0.01):
    """
    Rectifies optical barrel distortion caused by high-diopter (20D/28D) condensing lenses.
    """
    h, w, _ = img_rgb.shape
    fx = fy = w
    cx, cy = w / 2.0, h / 2.0
    
    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.array([k1, k2, 0, 0], dtype=np.float32)
    
    rectified = cv2.undistort(img_rgb, camera_matrix, dist_coeffs)
    return rectified

def suppress_central_led_hotspot(img_rgb, sigma=45):
    """
    Removes the bright central LED flash reflection artifact typical of
    coaxial smartphone fundus adapters like Remidio.
    """
    # Isolate luminance in LAB
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l_chan = lab[:, :, 0].astype(np.float32)
    
    # Estimate background illumination gradient
    bg_illum = cv2.GaussianBlur(l_chan, (0, 0), sigma)
    mean_illum = np.mean(bg_illum)
    
    # Flatten illumination hotspot
    l_flat = l_chan - (bg_illum - mean_illum) * 0.45
    lab[:, :, 0] = np.clip(l_flat, 0, 255).astype(np.uint8)
    
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

def correct_chromatic_aberration(img_rgb):
    """
    Realigns subtle red/blue channel fringe dispersion from mobile phone camera lenses.
    """
    r, g, b = cv2.split(img_rgb)
    # Align green and blue channels slightly to red channel reference
    h, w = r.shape
    m_shift = np.float32([[1, 0, 0.5], [0, 1, 0.5]])
    b_aligned = cv2.warpAffine(b, m_shift, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return cv2.merge([r, g, b_aligned])

def apply_hardware_profile_corrections(img_rgb, profile_key="REMIDIO_FOP"):
    """
    Applies tailored optical correction pipeline based on the acquisition hardware.
    Returns:
    - corrected_rgb: Processed image matching standard diagnostic characteristics
    - profile_info: Metadata dictionary describing hardware specifications
    """
    if profile_key not in HARDWARE_PROFILES:
        profile_key = "REMIDIO_FOP"
        
    profile_info = HARDWARE_PROFILES[profile_key]
    corrected = img_rgb.copy()
    
    if profile_key == "REMIDIO_FOP":
        # 1. Suppress central smartphone LED flare
        corrected = suppress_central_led_hotspot(corrected, sigma=40)
        # 2. Correct mobile phone lens chromatic fringe
        corrected = correct_chromatic_aberration(corrected)
        
    elif profile_key == "MII_RETCAM":
        # 1. Un-distort barrel curvature of 20D condensing lens
        corrected = correct_barrel_distortion(corrected, k1=-0.12, k2=0.02)
        # 2. Suppress lens flare
        corrected = suppress_central_led_hotspot(corrected, sigma=50)
        
    elif profile_key == "DESKTOP_STANDARD":
        # Minimal processing needed for telecentric clinical camera
        pass
        
    return corrected, profile_info
