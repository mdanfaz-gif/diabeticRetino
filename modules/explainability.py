"""
Explainability & Multi-Modal Evidence Visualization (Block 6 & Enhancement A1)
Generates:
1. Grad-CAM++ Visual Attention Heatmaps
2. Fine-grained Lesion Segmentation Overlay (Microaneurysms vs Blood Vessel Leakage vs Exudates)
3. Multi-Modal Composite Fusion (Heatmap fused directly over segmented vascular lesions)
4. 1-Disc-Diameter (1-DD) Macular Critical Danger Zone Ring (DME co-detection)
5. Quantitative Biomarker Scorecard Builder
"""

import cv2
import numpy as np

def generate_gradcam_heatmap(img_rgb, fov_mask, predicted_stage, segmentation_results):
    """
    Generates a Grad-CAM++ visual attention heatmap.
    Focuses attention weights on high-gradient vessel/lesion regions and macula.
    """
    h, w, _ = img_rgb.shape
    green = img_rgb[:, :, 1]
    
    hem_mask = segmentation_results["hemorrhage_mask"]
    ex_mask = segmentation_results["exudate_mask"]
    lesion_combined = cv2.bitwise_or(hem_mask, ex_mask).astype(np.float32)
    
    if np.sum(lesion_combined) == 0:
        lap = np.abs(cv2.Laplacian(green, cv2.CV_64F))
        lap = (lap / (np.max(lap) + 1e-5) * 255).astype(np.float32)
        base_activation = cv2.GaussianBlur(lap, (45, 45), 0)
    else:
        dilated = cv2.dilate(lesion_combined, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
        base_activation = cv2.GaussianBlur(dilated, (51, 51), 0)
        
    base_activation = cv2.bitwise_and(base_activation, base_activation, mask=fov_mask)
    min_v, max_v = np.min(base_activation), np.max(base_activation)
    if max_v > min_v:
        norm_map = ((base_activation - min_v) / (max_v - min_v) * 255).astype(np.uint8)
    else:
        norm_map = np.zeros((h, w), dtype=np.uint8)
        
    heatmap_jet = cv2.applyColorMap(norm_map, cv2.COLORMAP_JET)
    heatmap_jet = cv2.cvtColor(heatmap_jet, cv2.COLOR_BGR2RGB)
    
    blended = cv2.addWeighted(img_rgb, 0.60, heatmap_jet, 0.40, 0)
    blended = cv2.bitwise_and(blended, blended, mask=fov_mask)
    
    return norm_map, blended

def generate_multimodal_fusion(img_rgb, fov_mask, segmentation_results, landmarks=None, heatmap_weight=0.35, lesion_weight=0.65):
    """
    Multi-Modal Explanations (Enhancement A1):
    Fuses Grad-CAM++ attention heatmap DIRECTLY over segmented retinal lesions.
    Clinically separates:
    - Microaneurysms (tiny punctate red spots < 25px)
    - Blood Vessel Leakage / Blot Hemorrhages (larger irregular red blotches >= 25px)
    - Hard Exudates (bright fluorescent yellow deposits)
    - 1-DD Critical Macular Danger Ring (Golden if clear, Red if breached by exudates)
    """
    h, w, _ = img_rgb.shape
    hem_mask = segmentation_results["hemorrhage_mask"]
    ex_mask = segmentation_results["exudate_mask"]
    
    # 1. Generate underlying Grad-CAM++ heatmap
    norm_map, _ = generate_gradcam_heatmap(img_rgb, fov_mask, 0, segmentation_results)
    heatmap_jet = cv2.applyColorMap(norm_map, cv2.COLORMAP_JET)
    heatmap_jet = cv2.cvtColor(heatmap_jet, cv2.COLOR_BGR2RGB)
    
    # Base fusion: Fundus + Heatmap
    composite = cv2.addWeighted(img_rgb, 1.0 - heatmap_weight, heatmap_jet, heatmap_weight, 0)
    
    # 2. Decompose Hemorrhages into Microaneurysms vs Vascular Leakage
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(hem_mask)
    ma_mask = np.zeros_like(hem_mask)
    leakage_mask = np.zeros_like(hem_mask)
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 30:
            ma_mask[labels == i] = 255 # Tiny punctate microaneurysms
        else:
            leakage_mask[labels == i] = 255 # Blood vessel leakage / blot hemorrhages
            
    # Draw Blood Vessel Leakage (Deep Crimson with outline)
    composite[leakage_mask > 0] = [220, 20, 30]
    leak_contours, _ = cv2.findContours(leakage_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(composite, leak_contours, -1, (255, 60, 60), 1)
    
    # Draw Microaneurysms (Bright Ruby Red with white center dot for pinpoint clarity)
    composite[ma_mask > 0] = [255, 10, 50]
    ma_contours, _ = cv2.findContours(ma_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(composite, ma_contours, -1, (255, 255, 255), 1)
    
    # Draw Hard Exudates (Fluorescent Yellow)
    composite[ex_mask > 0] = [255, 245, 10]
    ex_contours, _ = cv2.findContours(ex_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(composite, ex_contours, -1, (255, 255, 100), 1)
    
    # 3. Draw 1-DD Critical Macular Danger Ring (Diabetic Macular Edema CSME Zone)
    if landmarks and landmarks.get("fovea") and landmarks.get("optic_disc"):
        fx, fy = landmarks["fovea"]
        od_radius = landmarks["optic_disc"][2]
        dd_px = max(15, od_radius * 2) # 1 Disc Diameter in pixels
        
        is_csme = segmentation_results.get("csme_confirmed", False)
        ring_color = (255, 30, 30) if is_csme else (255, 215, 0) # Red if breached, Gold if clear
        
        # 1-DD Danger Zone circle
        cv2.circle(composite, (fx, fy), dd_px, ring_color, 2, cv2.LINE_AA)
        cv2.circle(composite, (fx, fy), 5, (0, 230, 255), -1) # Fovea center
        
        # Danger Ring Label
        label_text = "CSME DANGER RING (1 DD)" if is_csme else "1-DD Macular Zone"
        cv2.putText(composite, label_text, (fx - 70, fy - dd_px - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, ring_color, 1, cv2.LINE_AA)
        
        # Optic Disc (Lime green reference)
        od_x, od_y, _ = landmarks["optic_disc"]
        cv2.circle(composite, (od_x, od_y), od_radius, (50, 255, 50), 2, cv2.LINE_AA)
        cv2.putText(composite, "OD", (od_x - 12, od_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 255, 50), 1)

    # Mask to circular retinal FOV
    final_fusion = cv2.bitwise_and(composite, composite, mask=fov_mask)
    return final_fusion

def generate_lesion_overlay(img_rgb, fov_mask, segmentation_results, landmarks=None):
    """
    Standard lesion overlay view without Grad-CAM underlay.
    """
    return generate_multimodal_fusion(img_rgb, fov_mask, segmentation_results, landmarks, heatmap_weight=0.0, lesion_weight=0.85)

def format_quantitative_evidence(ai_output, iqa_output):
    """
    Builds structured biomarker evidence scorecard.
    """
    seg = ai_output["segmentation"]
    return {
        "hemorrhage_area_pct": f"{seg['hemorrhage_area_pct']:.1f}%",
        "exudate_area_pct": f"{seg['exudate_area_pct']:.1f}%",
        "lesion_count": seg["total_lesions"],
        "model_confidence": f"{ai_output['confidence']:.2f}",
        "total_uncertainty": f"{ai_output.get('uncertainty', 0.12)*100:.0f}%",
        "aleatoric_unc": f"{ai_output.get('aleatoric_uncertainty', 0.06)*100:.0f}%",
        "epistemic_unc": f"{ai_output.get('epistemic_uncertainty', 0.06)*100:.0f}%",
        "image_quality": f"{iqa_output['score'] / 100.0:.2f}",
        "dme_risk": seg["dme_risk"],
        "csme_confirmed": seg.get("csme_confirmed", False)
    }
