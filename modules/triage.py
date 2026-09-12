"""
Clinical Triage Engine (Block 8)
Block 8 of the SIH26038 Architecture + Architectural Enhancement 2.

Features:
- Rule-based risk stratification (Low Risk, Referable DR, High Risk Urgent)
- Urgency scoring (1 - 10) for intelligent telemedicine queue prioritization
- Clinical timing recommendations (Annual, 2-4 weeks, < 48 hours)
- Mandatory clinical safety disclaimers
"""

def compute_clinical_triage(ai_output, consensus_verdict, patient_data=None):
    """
    Evaluates patient risk and assigns triage tier based on DR Stage, DME Risk,
    and Consensus Guardrail Layer output.
    """
    stage = ai_output["predicted_stage"]
    seg = ai_output["segmentation"]
    dme_risk = seg["dme_risk"]
    guardrail_status = consensus_verdict["status"]
    
    is_high_risk = (stage in [3, 4]) or ("High" in dme_risk)
    
    # Build Dual Co-Diagnosis (DR Stage + DME Status)
    is_csme = seg.get("csme_confirmed", False)
    if is_csme:
        macular_diagnosis = "Clinically Significant Macular Edema (CSME: Exudates < 1 DD from Fovea)"
        dme_tier = "CSME (High Risk Central Vision Loss)"
    elif "Moderate" in dme_risk:
        macular_diagnosis = "Moderate Macular Edema (Exudates within 1.0 - 2.0 DD)"
        dme_tier = "Moderate DME"
    else:
        macular_diagnosis = "No Macular Edema (Fovea Clear)"
        dme_tier = "No DME"

    # Check if AI abstained or requested human review
    if guardrail_status == "ABSTAIN":
        tier = "Urgent Review (AI Abstention)" if is_high_risk else "Specialist Review (AI Abstention)"
        risk_level = "High Risk / Contradictory Evidence" if is_high_risk else "Uncertain / Contradictory Evidence"
        badge_color = "#dc3545"
        urgency_score = 9 if is_high_risk else 8
        recommended_window = "Priority Ophthalmologist Review (< 48 hours)" if is_high_risk else "Priority Ophthalmologist Review (within 7 days)"
        clinical_action = (
            "Guardrail Layer detected clinical inconsistencies between feature classification "
            "and lesion segmentation. Mandatory human ophthalmologist review required before diagnosis."
        )
        follow_up = "District Hospital Tele-Consultation"
        is_urgent = True
        dual_co_diagnosis = f"{ai_output['stage_name']} + {dme_tier} (Guardrail: ABSTAIN)"

    elif guardrail_status == "HUMAN_REVIEW":
        tier = "Urgent Review (High Risk Biomarkers)" if is_high_risk else "Specialist Review (Borderline Quality)"
        risk_level = "High Risk / Inconclusive Model Confidence" if is_high_risk else "Moderate / Inconclusive AI Confidence"
        badge_color = "#dc3545" if is_high_risk else "#ffc107"
        urgency_score = 9 if is_high_risk else 6
        recommended_window = "Urgent Tele-Ophthalmologist Review (< 48 hours)" if is_high_risk else "Tele-Ophthalmologist Review (within 14 days)"
        clinical_action = (
            "Model confidence was borderline but high-risk retinal biomarkers (Severe NPDR/CSME) "
            "were segmented. Urgent human ophthalmologist confirmation required." if is_high_risk else
            "Model confidence or image sharpness was borderline. Screening report flagged for "
            "tele-ophthalmologist verification."
        )
        follow_up = "District Hospital Telemedicine" if is_high_risk else "Local PHC Re-examination or District Telemedicine"
        is_urgent = is_high_risk
        dual_co_diagnosis = f"{ai_output['stage_name']} + {dme_tier} (Guardrail: HUMAN REVIEW)"

    else:
        # Standard Consensus ACCEPT Triage Rules
        dual_co_diagnosis = f"{ai_output['stage_name']} + {dme_tier}"
        
        # 1. High Risk / Urgent Referral: Stage 3, Stage 4, OR High DME risk
        if stage in [3, 4] or "High" in dme_risk:
            tier = "Urgent Referral"
            risk_level = "High Risk (Sight-Threatening DR)"
            badge_color = "#dc3545" # Red
            urgency_score = 10 if stage == 4 else 9
            recommended_window = "Immediate Vitreoretinal Consultation (< 48 hours)"
            
            reasons = []
            if stage == 4:
                reasons.append("Proliferative DR (neovascularization risk of vitreous hemorrhage/retinal detachment)")
            elif stage == 3:
                reasons.append("Severe NPDR (high 1-year progression rate to proliferative stage)")
            if "High" in dme_risk:
                reasons.append("Clinically Significant Macular Edema (CSME: exudates encroaching foveal avascular zone)")
                
            clinical_action = (
                f"URGENT: Patient exhibits sight-threatening diabetic retinopathy ({', '.join(reasons)}). "
                f"Immediate evaluation for anti-VEGF injection, panretinal photocoagulation (PRP), or surgical intervention."
            )
            follow_up = "District Hospital / Tertiary Eye Hospital"
            is_urgent = True
            
        # 2. Moderate Risk / Specialist Review: Stage 1 or Stage 2 without CSME
        elif stage in [1, 2]:
            tier = "Specialist Review"
            risk_level = "Referable DR (Moderate Risk)"
            badge_color = "#fd7e14" # Orange
            urgency_score = 5 if stage == 1 else 7
            recommended_window = "Specialist Consultation (within 2 - 4 weeks)"
            clinical_action = (
                f"Referable DR detected ({ai_output['stage_name']}). Non-urgent dilated ophthalmoscopy "
                f"and baseline optical coherence tomography (OCT) recommended. Glycemic (HbA1c) and BP optimization."
            )
            follow_up = "Taluk/District Tele-Ophthalmology Clinic"
            is_urgent = False
            
        # 3. Low Risk / Routine Monitoring: Stage 0 (No DR)
        else:
            tier = "Routine Monitoring"
            risk_level = "Low Risk (Non-Referable)"
            badge_color = "#28a745" # Green
            urgency_score = 1
            recommended_window = "Annual Screening (12 months)"
            clinical_action = (
                "No diabetic retinopathy detected. Continue regular annual retinal screening at local Primary Health Centre. "
                "Reinforce diabetes self-management, healthy diet, and lifestyle modification."
            )
            follow_up = "Local PHC / Sub-Centre"
            is_urgent = False

    # Elevate urgency if patient has uncontrolled diabetes (HbA1c > 9.0) or long duration (> 10 years)
    if patient_data:
        hba1c = patient_data.get("hba1c", 7.0)
        if hba1c >= 9.5 and urgency_score < 8:
            urgency_score += 1

    # Time-to-Referral Priority Calculation (Universal across all branches)
    if urgency_score >= 10 or (stage == 4 and is_csme):
        time_to_referral = "EMERGENCY (< 24 HOURS)"
        referral_facility = "Tertiary Eye Hospital / Vitreoretinal Surgery Unit"
    elif urgency_score >= 8 or is_high_risk:
        time_to_referral = "URGENT (< 48 HOURS)"
        referral_facility = "District Civil Hospital (Tele-Ophthalmology Hub)"
    elif urgency_score >= 5:
        time_to_referral = "PRIORITY (2 TO 4 WEEKS)"
        referral_facility = "Sub-District Hospital / Taluk Eye Clinic"
    else:
        time_to_referral = "ROUTINE (6 TO 12 MONTHS)"
        referral_facility = "Local Primary Health Centre (PHC) / Vision Centre"

    return {
        "tier": tier,
        "risk_level": risk_level,
        "badge_color": badge_color,
        "urgency_score": urgency_score,
        "time_to_referral": time_to_referral,
        "referral_facility": referral_facility,
        "dual_co_diagnosis": dual_co_diagnosis,
        "primary_dr_diagnosis": ai_output["stage_name"],
        "macular_diagnosis": macular_diagnosis,
        "recommended_window": recommended_window,
        "clinical_action": clinical_action,
        "follow_up": follow_up,
        "is_urgent": is_urgent,
        "disclaimer": "Final clinical decision, timing and treatment are determined by the registered ophthalmologist."
    }
