"""
Curated Realistic Fundus Sample Cases Generator
Enables instant zero-download interactive testing during hackathon presentations.

Synthesizes high-fidelity 512x512 fundus images matching standard 45-degree field:
1. Normal Retinal Fundus (Stage 0)
2. Mild NPDR (Stage 1: Isolated microaneurysms)
3. Moderate NPDR (Stage 2: Exudates + hemorrhages)
4. Severe NPDR with CSME (Stage 3: Dense hemorrhages + Exudates < 1 DD from fovea)
5. Proliferative DR (Stage 4: Neovascular fronds + extensive blot hemorrhages)
6. Low-Quality Blurry Fundus (Triggers IQA Recapture Gate)
7. Diagnostic Contradiction Edge Case (Triggers Multi-Agent ABSTAIN Guardrail)
"""

import cv2
import numpy as np
import os

SAMPLE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_cases")

def create_base_retina(width=512, height=512, eye="OD"):
    """
    Creates realistic retinal background:
    - Circular fundus boundary with dark vignetting
    - Realistic orange-red choroid color gradient
    - Optic Disc (yellowish-white circular region with cup)
    - Fovea / Macula (darker avascular zone)
    - Retinal vascular tree
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)
    center_x, center_y = width // 2, height // 2
    retina_radius = int(width * 0.46)
    
    # 1. Circular Eyeball FOV Mask
    y_coords, x_coords = np.ogrid[:height, :width]
    dist_from_center = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
    fov_mask = (dist_from_center <= retina_radius).astype(np.uint8) * 255
    
    # 2. Base Choroidal Gradient (Red-Orange Hue: R~195-215, G~75-95, B~25-35)
    falloff = np.clip(1.0 - (dist_from_center / retina_radius)**2 * 0.35, 0.5, 1.0)
    
    img[:, :, 0] = (205 * falloff).astype(np.uint8)
    img[:, :, 1] = (85 * falloff).astype(np.uint8)
    img[:, :, 2] = (25 * falloff).astype(np.uint8)
    
    # Smooth slight Gaussian to avoid artificial sharp noise
    img = cv2.GaussianBlur(img, (5, 5), 0)
    
    # 3. Optic Disc Position
    od_x = int(width * 0.74) if eye == "OD" else int(width * 0.26)
    od_y = int(height * 0.50)
    od_r = int(width * 0.075)
    
    # Draw Optic Disc (outer rim & cup)
    cv2.circle(img, (od_x, od_y), od_r, (235, 175, 90), -1)
    cv2.circle(img, (od_x, od_y), int(od_r * 0.5), (250, 220, 150), -1)
    
    # 4. Fovea / Macula Position
    fovea_x = int(width * 0.42) if eye == "OD" else int(width * 0.58)
    fovea_y = int(height * 0.52)
    fovea_r = int(width * 0.05)
    
    # Darker avascular zone with smooth falloff
    macula_mask = np.zeros((height, width), dtype=np.float32)
    cv2.circle(macula_mask, (fovea_x, fovea_y), fovea_r, 1.0, -1)
    macula_mask_blur = cv2.GaussianBlur(macula_mask, (45, 45), 0)
    
    for c, factor in enumerate([0.15, 0.28, 0.35]):
        ch = img[:, :, c].astype(np.float32)
        ch = ch * (1.0 - macula_mask_blur * factor)
        img[:, :, c] = np.clip(ch, 0, 255).astype(np.uint8)
    
    # 5. Retinal Blood Vessels (Arching gracefully from Optic Disc)
    vessel_color = (120, 30, 15) # Dark red / burgundy
    
    pts_sup = np.array([
        [od_x, od_y],
        [od_x - 40 if eye == "OD" else od_x + 40, od_y - 80],
        [fovea_x + 30 if eye == "OD" else fovea_x - 30, od_y - 120],
        [fovea_x - 50 if eye == "OD" else fovea_x + 50, od_y - 100],
        [int(width * 0.2) if eye == "OD" else int(width * 0.8), od_y - 70]
    ], np.int32)
    
    pts_inf = np.array([
        [od_x, od_y],
        [od_x - 40 if eye == "OD" else od_x + 40, od_y + 80],
        [fovea_x + 30 if eye == "OD" else fovea_x - 30, od_y + 120],
        [fovea_x - 50 if eye == "OD" else fovea_x + 50, od_y + 100],
        [int(width * 0.2) if eye == "OD" else int(width * 0.8), od_y + 70]
    ], np.int32)
    
    cv2.polylines(img, [pts_sup], False, vessel_color, 4, cv2.LINE_AA)
    cv2.polylines(img, [pts_inf], False, vessel_color, 4, cv2.LINE_AA)
    
    nasal_dir = 1 if eye == "OD" else -1
    cv2.line(img, (od_x, od_y), (od_x + int(nasal_dir * 80), od_y - 60), vessel_color, 3, cv2.LINE_AA)
    cv2.line(img, (od_x, od_y), (od_x + int(nasal_dir * 80), od_y + 60), vessel_color, 3, cv2.LINE_AA)
    
    # Mask strictly to circular FOV
    img = cv2.bitwise_and(img, img, mask=fov_mask)
    return img, (od_x, od_y, od_r), (fovea_x, fovea_y), fov_mask

def add_microaneurysms(img, count=4, seed=101):
    """Adds tiny punctate red dots (Microaneurysms, 2-3px radius)."""
    np.random.seed(seed)
    h, w, _ = img.shape
    for _ in range(count):
        rx = np.random.randint(int(w * 0.25), int(w * 0.65))
        ry = np.random.randint(int(h * 0.3), int(h * 0.7))
        cv2.circle(img, (rx, ry), np.random.randint(2, 4), (80, 10, 8), -1)
    return img

def add_blot_hemorrhages(img, count=12, seed=202):
    """Adds irregular deep red blotches (Hemorrhages, 5-14px radius)."""
    np.random.seed(seed)
    h, w, _ = img.shape
    for _ in range(count):
        rx = np.random.randint(int(w * 0.2), int(w * 0.75))
        ry = np.random.randint(int(h * 0.2), int(h * 0.8))
        r1, r2 = np.random.randint(4, 8), np.random.randint(5, 12)
        angle = np.random.randint(0, 180)
        cv2.ellipse(img, (rx, ry), (r1, r2), angle, 0, 360, (70, 10, 8), -1)
    return img

def add_hard_exudates(img, count=16, near_fovea=False, fovea_pos=(215, 266), seed=303):
    """Adds bright waxy yellowish-white lipid exudate clusters."""
    np.random.seed(seed)
    h, w, _ = img.shape
    fx, fy = fovea_pos
    
    for _ in range(count):
        if near_fovea:
            # Clustered tightly within 0.8 Disc Diameter (<45px) of Fovea (CSME / DME)
            rx = fx + np.random.randint(-35, 35)
            ry = fy + np.random.randint(-35, 35)
        else:
            # Distant from fovea: at least 120px (>2.5 Disc Diameters) away in temporal or nasal periphery
            rx = np.random.choice([np.random.randint(int(w * 0.15), fx - 70), np.random.randint(fx + 70, int(w * 0.85))])
            ry = np.random.randint(int(h * 0.2), int(h * 0.8))
            
        r = np.random.randint(2, 5)
        cv2.circle(img, (rx, ry), r, (252, 242, 170), -1)
    return img

def generate_sample_cases():
    """Generates and saves all 7 curated test cases into sample_cases/."""
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    
    # 1. Normal (Stage 0) - pristine fundus, 0 lesions
    normal_img, od, fov, mask = create_base_retina()
    cv2.imwrite(os.path.join(SAMPLE_DIR, "case1_normal_stage0.jpg"), cv2.cvtColor(normal_img, cv2.COLOR_RGB2BGR))
    
    # 2. Mild NPDR (Stage 1) - 4 microaneurysms only
    mild_img, _, _, _ = create_base_retina()
    mild_img = add_microaneurysms(mild_img, count=4, seed=42)
    cv2.imwrite(os.path.join(SAMPLE_DIR, "case2_mild_stage1.jpg"), cv2.cvtColor(mild_img, cv2.COLOR_RGB2BGR))
    
    # 3. Moderate NPDR (Stage 2) - scattered exudates (distant from fovea) + moderate hemorrhages
    mod_img, _, fov_pos, _ = create_base_retina()
    mod_img = add_microaneurysms(mod_img, count=6, seed=55)
    mod_img = add_blot_hemorrhages(mod_img, count=5, seed=56)
    mod_img = add_hard_exudates(mod_img, count=8, near_fovea=False, fovea_pos=fov_pos, seed=57)
    cv2.imwrite(os.path.join(SAMPLE_DIR, "case3_moderate_stage2.jpg"), cv2.cvtColor(mod_img, cv2.COLOR_RGB2BGR))
    
    # 4. Severe NPDR with High DME (Stage 3 + CSME) - 20 hemorrhages + exudates < 1 DD from fovea
    severe_img, _, fov_pos, _ = create_base_retina()
    severe_img = add_blot_hemorrhages(severe_img, count=22, seed=77)
    severe_img = add_hard_exudates(severe_img, count=20, near_fovea=True, fovea_pos=fov_pos, seed=78)
    cv2.imwrite(os.path.join(SAMPLE_DIR, "case4_severe_csme_stage3.jpg"), cv2.cvtColor(severe_img, cv2.COLOR_RGB2BGR))
    
    # 5. Proliferative DR (Stage 4) - 35+ blot hemorrhages + neovascular fronds
    pdr_img, od_pos, fov_pos, _ = create_base_retina()
    pdr_img = add_blot_hemorrhages(pdr_img, count=36, seed=99)
    pdr_img = add_hard_exudates(pdr_img, count=24, near_fovea=False, fovea_pos=fov_pos, seed=100)
    od_x, od_y, _ = od_pos
    for _ in range(8):
        dx, dy = np.random.randint(-25, 25), np.random.randint(-25, 25)
        cv2.line(pdr_img, (od_x, od_y), (od_x + dx, od_y + dy), (120, 20, 10), 2, cv2.LINE_AA)
    cv2.imwrite(os.path.join(SAMPLE_DIR, "case5_proliferative_stage4.jpg"), cv2.cvtColor(pdr_img, cv2.COLOR_RGB2BGR))
    
    # 6. Low Quality Blurry Fundus (Triggers IQA Gate Rejection)
    blur_img, _, _, _ = create_base_retina()
    blur_img = cv2.GaussianBlur(blur_img, (31, 31), 15)
    cv2.imwrite(os.path.join(SAMPLE_DIR, "case6_blurry_iqa_reject.jpg"), cv2.cvtColor(blur_img, cv2.COLOR_RGB2BGR))
    
    # 7. Contradiction Edge Case (Triggers Multi-Agent ABSTAIN)
    contra_img, _, fov_pos, _ = create_base_retina()
    contra_img = add_hard_exudates(contra_img, count=18, near_fovea=False, fovea_pos=fov_pos, seed=888)
    cv2.imwrite(os.path.join(SAMPLE_DIR, "case7_edge_case_contradiction.jpg"), cv2.cvtColor(contra_img, cv2.COLOR_RGB2BGR))
    
    print(f"Successfully generated 7 curated test cases in: {SAMPLE_DIR}")

if __name__ == "__main__":
    generate_sample_cases()
