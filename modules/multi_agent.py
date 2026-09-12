"""
Multi-Agent Analysis & Consensus Guardrail Layer (Block 5 & 7)
Block 5 & 7 of the SIH26038 Architecture + Architectural Enhancement 3.

Agents:
- Agent 1: Vision Analyst (Extracts quantitative vision features, probabilities, attention maps)
- Agent 2: Clinical Validator (Neuro-Symbolic ETDRS/ICDR rule engine, contradiction detector)
- Consensus & Guardrail Engine (Stateflow / Simulink logic: ACCEPT, ABSTAIN, HUMAN REVIEW)
"""

import json

class VisionAnalystAgent:
    """
    Agent 1: Vision Analyst
    Synthesizes deep model outputs and lesion segmentation into structured clinical evidence.
    """
    def __init__(self):
        self.agent_id = "Agent_1_VisionAnalyst"
        self.role = "Deep Retinal Feature & Lesion Quantification Specialist"
        
    def analyze(self, ai_output, iqa_output):
        """
        Formulates structured clinical evidence payload from computer vision models.
        """
        seg = ai_output["segmentation"]
        
        evidence = {
            "source_agent": self.agent_id,
            "role": self.role,
            "dr_stage_prediction": ai_output["predicted_stage"],
            "dr_stage_name": ai_output["stage_name"],
            "confidence": ai_output["confidence"],
            "epistemic_uncertainty": ai_output["uncertainty"],
            "iqa_score": iqa_output["score"],
            "quantitative_biomarkers": {
                "hemorrhage_count": seg["hemorrhage_count"],
                "hemorrhage_area_pct": seg["hemorrhage_area_pct"],
                "exudate_count": seg["exudate_count"],
                "exudate_area_pct": seg["exudate_area_pct"],
                "total_lesion_count": seg["total_lesions"],
                "dme_risk": seg["dme_risk"],
                "min_foveal_distance_dd": seg["min_dist_fovea_dd"]
            },
            "summary_rationale": (
                f"Predicted {ai_output['stage_name']} with confidence {ai_output['confidence']*100:.1f}% "
                f"(uncertainty={ai_output['uncertainty']:.2f}). Detected {seg['total_lesions']} lesions "
                f"({seg['hemorrhage_area_pct']}% hemorrhages, {seg['exudate_area_pct']}% exudates). "
                f"Macular status: {seg['dme_risk']}."
            )
        }
        return evidence

class ClinicalValidatorAgent:
    """
    Agent 2: Clinical Validator
    Applies an explicit Neuro-Symbolic Rule Base (ICDR / ETDRS standards)
    to cross-verify AI vision findings against medical diagnostic criteria.
    """
    def __init__(self):
        self.agent_id = "Agent_2_ClinicalValidator"
        self.role = "Ophthalmology Rule-Based Clinical Auditor (ETDRS Compliant)"
        
    def validate(self, vision_evidence):
        """
        Executes clinical rule checks against Vision Analyst evidence.
        Detects medical contradictions, unverified claims, and calculates validation score.
        """
        stage = vision_evidence["dr_stage_prediction"]
        bm = vision_evidence["quantitative_biomarkers"]
        conf = vision_evidence["confidence"]
        unc = vision_evidence["epistemic_uncertainty"]
        
        rules_passed = []
        contradictions = []
        warnings = []
        
        # Rule 1: ETDRS Stage 0 (No DR) Rule
        if stage == 0:
            if bm["hemorrhage_count"] == 0 and bm["exudate_count"] == 0:
                rules_passed.append("ETDRS-0: Absence of microaneurysms, hemorrhages, and exudates confirmed.")
            else:
                contradictions.append(
                    f"ETDRS Contradiction: Stage 0 (No DR) claimed, but {bm['total_lesion_count']} "
                    f"lesions ({bm['hemorrhage_count']} hemorrhages, {bm['exudate_count']} exudates) were segmented."
                )
                
        # Rule 2: ETDRS Stage 1 (Mild NPDR) Rule: Microaneurysms ONLY
        elif stage == 1:
            if bm["exudate_count"] > 0:
                contradictions.append(
                    f"ETDRS Contradiction: Mild NPDR permits microaneurysms ONLY. "
                    f"Exudates ({bm['exudate_count']} detected) require at least Stage 2 (Moderate NPDR)."
                )
            elif bm["hemorrhage_count"] > 10:
                warnings.append("High microaneurysm count for Mild NPDR; borderline Moderate NPDR.")
            else:
                rules_passed.append("ETDRS-1: Isolated microaneurysms without exudates verified.")
                
        # Rule 3: ETDRS Stage 2 (Moderate NPDR) Rule
        elif stage == 2:
            if bm["total_lesion_count"] == 0:
                contradictions.append("ETDRS Contradiction: Moderate NPDR claimed, but zero lesions detected.")
            else:
                rules_passed.append("ETDRS-2: Lesion profile congruent with Moderate NPDR (exudates and/or moderate hemorrhages).")
                
        # Rule 4: ETDRS Stage 3 (Severe NPDR) Rule: 4-2-1 Rule
        elif stage == 3:
            if bm["hemorrhage_area_pct"] < 0.2 and bm["hemorrhage_count"] < 8:
                contradictions.append("ETDRS Contradiction: Insufficient hemorrhage burden for Severe NPDR.")
            else:
                rules_passed.append("ETDRS-3: Multi-quadrant severe hemorrhage burden verified.")
                
        # Rule 5: ETDRS Stage 4 (Proliferative DR) Rule
        elif stage == 4:
            if bm["total_lesion_count"] < 5 and bm["hemorrhage_area_pct"] < 0.5:
                contradictions.append("ETDRS Contradiction: Proliferative DR claimed, but lesion density is negligible.")
            else:
                rules_passed.append("ETDRS-4: High-density proliferative lesion profile verified.")
                
        # Rule 6: Clinically Significant Macular Edema (CSME) Rule
        if "High" in bm["dme_risk"]:
            warnings.append("URGENT: Hard exudates detected within 1 Disc Diameter of Fovea. Macular edema risk high.")
            
        # Compute Clinical Validation Score (0.0 - 1.0)
        validation_score = 1.0
        if contradictions:
            validation_score -= 0.40 * len(contradictions)
        if warnings:
            validation_score -= 0.05 * len(warnings)
        if unc > 0.65:
            validation_score -= 0.15
            
        validation_score = max(0.05, min(1.0, validation_score))
        
        return {
            "source_agent": self.agent_id,
            "role": self.role,
            "clinical_validation_score": round(validation_score, 2),
            "rules_passed": rules_passed,
            "contradictions": contradictions,
            "warnings": warnings,
            "clinical_verdict": "CONGRUENT" if not contradictions else "CONTRADICTORY",
            "summary_rationale": (
                f"Clinical Validation Score: {validation_score:.2f}. "
                f"{'All ETDRS rules verified.' if not contradictions else 'Contradictions detected against medical staging rules.'}"
            )
        }

# -------------------------------------------------------------
# Block 7: Consensus & Guardrail Layer (Stateflow / Simulink Logic)
# -------------------------------------------------------------
def evaluate_consensus_guardrail(vision_evidence, clinical_validation):
    """
    Compares Agent 1 & Agent 2 outputs using Stateflow / Simulink Guardrail Logic:
    
    1. ACCEPT (High Confidence Consensus):
       - Vision confidence >= 0.60
       - Clinical validation score >= 0.65
       - Zero contradictions
       -> Action: Proceed to automated triage.
       
    2. ABSTAIN (Contradictory Evidence):
       - Clinical contradictions detected between vision classification and lesion segmentation
       -> Action: Flag for Senior Ophthalmologist Review with reason 'Contradictory Evidence'.
       
    3. HUMAN REVIEW (Insufficient Evidence / High Uncertainty):
       - Confidence < 0.60, or uncertainty > 0.65, or IQA score < 65
       -> Action: Route to tele-ophthalmologist with reason 'High Uncertainty / Borderline Image Quality'.
    """
    conf = vision_evidence["confidence"]
    unc = vision_evidence["epistemic_uncertainty"]
    iqa = vision_evidence["iqa_score"]
    val_score = clinical_validation["clinical_validation_score"]
    contradictions = clinical_validation["contradictions"]
    
    if len(contradictions) > 0 or clinical_validation["clinical_verdict"] == "CONTRADICTORY":
        status = "ABSTAIN"
        decision_code = "DISAGREE_CONTRADICTION"
        description = "Contradictory evidence detected between classifier and lesion segmentation."
        action = "Flag for Senior Ophthalmologist Review (AI Abstention Safety Triggered)"
        badge_color = "#dc3545" # Red
        
    elif conf < 0.58 or unc > 0.65 or iqa < 62.0 or val_score < 0.65:
        status = "HUMAN_REVIEW"
        decision_code = "LOW_CONFIDENCE_UNCERTAINTY"
        description = "Insufficient evidence or high epistemic uncertainty detected."
        action = "Escalate to Tele-Ophthalmologist for Manual Assessment"
        badge_color = "#ffc107" # Yellow
        
    else:
        status = "ACCEPT"
        decision_code = "AGREE_HIGH_CONFIDENCE"
        description = "Full consensus between Vision Analyst and Clinical Validator."
        action = "Proceed directly to Clinical Triage Engine"
        badge_color = "#28a745" # Green
        
    return {
        "status": status,
        "decision_code": decision_code,
        "description": description,
        "action": action,
        "badge_color": badge_color,
        "consensus_metrics": {
            "vision_confidence": conf,
            "epistemic_uncertainty": unc,
            "clinical_validation_score": val_score,
            "iqa_score": iqa,
            "contradiction_count": len(contradictions)
        }
    }
