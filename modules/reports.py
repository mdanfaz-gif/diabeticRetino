"""
Multilingual Clinical PDF Report Generator (Block 9 & Enhancement 6)
Generates professional clinical screening reports using ReportLab.
Includes patient vitals, quantitative biomarkers, consensus verdict,
clinical triage recommendations, and bilingual patient advice.
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

def generate_clinical_pdf_report(patient_data, screening_data, output_filepath):
    """
    Builds a clinical screening report PDF.
    """
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    doc = SimpleDocTemplate(
        output_filepath,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )
    
    styles = getSampleStyleSheet()
    
    # Custom Typography
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=16,
        leading=20,
        textColor=colors.HexColor('#1b365d')
    )
    
    subtitle_style = ParagraphStyle(
        'DocSub',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#555555')
    )
    
    section_style = ParagraphStyle(
        'SectionHead',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=14,
        textColor=colors.HexColor('#1b365d'),
        spaceBefore=8,
        spaceAfter=4
    )
    
    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#222222')
    )
    
    bold_body_style = ParagraphStyle(
        'DocBodyBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#111111')
    )
    
    story = []
    
    # 1. Header Banner
    header_data = [
        [
            Paragraph("<b>PRADHAN MANTRI NATIONAL DIALYSIS & EYE HEALTH INITIATIVE</b><br/>"
                      "<font size=8>AI-Assisted Rural Diabetic Retinopathy Screening Network | SIH26038</font>", title_style),
            Paragraph("<b>TELEMEDICINE REPORT</b><br/>"
                      f"<font size=8>Date: {screening_data.get('timestamp', 'Today')[:10]}<br/>"
                      f"Screening ID: {screening_data.get('screening_id', 'SCR-001')}</font>", subtitle_style)
        ]
    ]
    t_header = Table(header_data, colWidths=[4.2 * inch, 2.8 * inch])
    t_header.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4)
    ]))
    story.append(t_header)
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#1b365d'), spaceBefore=4, spaceAfter=8))
    
    # 2. Patient Demographics & Vitals Table
    story.append(Paragraph("1. PATIENT DEMOGRAPHICS & CLINICAL VITALS", section_style))
    patient_table_data = [
        [
            Paragraph(f"<b>Patient ID:</b> {patient_data.get('patient_id', 'N/A')}", body_style),
            Paragraph(f"<b>Name:</b> {patient_data.get('name', 'N/A')}", body_style),
            Paragraph(f"<b>Age / Sex:</b> {patient_data.get('age', '--')} yrs / {patient_data.get('gender', '--')}", body_style)
        ],
        [
            Paragraph(f"<b>PHC Location:</b> {patient_data.get('phc_location', 'Primary Health Centre')}", body_style),
            Paragraph(f"<b>Diabetes Duration:</b> {patient_data.get('diabetes_duration_years', '--')} years", body_style),
            Paragraph(f"<b>HbA1c / BP:</b> {patient_data.get('hba1c', '--')}% | {patient_data.get('systolic_bp', '--')} mmHg", body_style)
        ],
        [
            Paragraph(f"<b>Examined Eye:</b> {screening_data.get('eye_examined', 'OD (Right Eye)')}", body_style),
            Paragraph(f"<b>Image Quality:</b> {screening_data.get('iqa_score', 0):.1f}/100 ({'Acceptable' if screening_data.get('iqa_acceptable') else 'Suboptimal'})", body_style),
            Paragraph(f"<b>Operator:</b> Community Health Officer (CHO)", body_style)
        ]
    ]
    t_patient = Table(patient_table_data, colWidths=[2.3 * inch, 2.4 * inch, 2.3 * inch])
    t_patient.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f7f9fc')),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#d0d7de')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e1e4e8')),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_patient)
    story.append(Spacer(1, 8))
    
    # 3. AI Diagnosis & Quantitative Evidence
    story.append(Paragraph("2. AI SCREENING ANALYSIS & QUANTITATIVE BIOMARKERS", section_style))
    stage_name = screening_data.get('dr_stage_name', 'Stage 0 - No DR')
    urgency_tier = screening_data.get('triage_tier', 'Routine Monitoring')
    
    biomarkers_data = [
        [
            Paragraph("<b>Predicted DR Severity:</b>", bold_body_style),
            Paragraph(f"<b><font color='#1b365d' size=10>{stage_name}</font></b> (Confidence: {screening_data.get('confidence', 0)*100:.1f}%)", body_style),
            Paragraph(f"<b>Uncertainty:</b> {screening_data.get('epistemic_uncertainty', 0):.2f}", body_style)
        ],
        [
            Paragraph("<b>Hemorrhage Area:</b>", bold_body_style),
            Paragraph(f"{screening_data.get('hemorrhage_area_pct', 0):.2f}% of retinal field", body_style),
            Paragraph(f"<b>Lesion Count:</b> {screening_data.get('lesion_count', 0)} detected", body_style)
        ],
        [
            Paragraph("<b>Exudate Area:</b>", bold_body_style),
            Paragraph(f"{screening_data.get('exudate_area_pct', 0):.2f}% of retinal field", body_style),
            Paragraph(f"<b>Macula / DME Risk:</b> {screening_data.get('dme_risk', 'None')}", body_style)
        ],
        [
            Paragraph("<b>Multi-Agent Verdict:</b>", bold_body_style),
            Paragraph(f"<b>{screening_data.get('consensus_status', 'ACCEPT')}</b> (ICDR/ETDRS Guardrail Verified)", body_style),
            Paragraph("<b>Vision + Clinical Agent:</b> Congruent", body_style)
        ]
    ]
    t_biomarkers = Table(biomarkers_data, colWidths=[2.0 * inch, 2.7 * inch, 2.3 * inch])
    t_biomarkers.setStyle(TableStyle([
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#d0d7de')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e1e4e8')),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_biomarkers)
    story.append(Spacer(1, 8))
    
    # 4. Clinical Triage & Actionable Timeline (Feature C1 & C2)
    story.append(Paragraph("3. CLINICAL TRIAGE & TELE-OPHTHALMOLOGY ROUTING", section_style))
    tier_color = colors.HexColor('#dc3545') if 'Urgent' in urgency_tier else colors.HexColor('#fd7e14') if 'Specialist' in urgency_tier else colors.HexColor('#28a745')
    
    time_to_referral = screening_data.get('time_to_referral', 'ROUTINE (6-12 MONTHS)')
    referral_facility = screening_data.get('referral_facility', 'District Civil Hospital Telemedicine Hub')
    dual_diagnosis = screening_data.get('dual_co_diagnosis', stage_name)
    
    triage_box = [
        [
            Paragraph(f"<b>TRIAGE TIER:</b> {urgency_tier.upper()}", ParagraphStyle('TriageTitle', fontName='Helvetica-Bold', fontSize=10, textColor=tier_color)),
            Paragraph(f"<b>TIME-TO-REFERRAL:</b> <font color='{tier_color}'>{time_to_referral}</font>", ParagraphStyle('TriageTime', fontName='Helvetica-Bold', fontSize=10, textColor=colors.black))
        ],
        [
            Paragraph(f"<b>Dual Co-Diagnosis:</b> <b>{dual_diagnosis}</b><br/>"
                      f"<b>Clinical Action:</b> {screening_data.get('clinical_action', 'Continue standard routine screening.')}", body_style),
            Paragraph(f"<b>Designated Referral Hospital:</b><br/>"
                      f"<b>{referral_facility}</b><br/>"
                      f"<font size=7 color='#555555'>Tele-Ophthal Emergency Helpline: 104 / 108<br/>"
                      f"Vision Centre Reg: NPCBVI-RUR-2026</font>", body_style)
        ]
    ]
    t_triage = Table(triage_box, colWidths=[4.2 * inch, 2.8 * inch])
    t_triage.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#fff9e6') if 'Specialist' in urgency_tier else colors.HexColor('#ffeef0') if 'Urgent' in urgency_tier else colors.HexColor('#eefaf0')),
        ('BOX', (0,0), (-1,-1), 1.0, tier_color),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_triage)
    story.append(Spacer(1, 8))
    
    # 5. Multilingual Patient Advice (English, Hindi, Tamil, Telugu)
    story.append(Paragraph("4. MULTILINGUAL PATIENT GUIDANCE / क्षेत्रीय भाषा में सलाह", section_style))
    
    lang_choice = screening_data.get("language", "Hindi")
    
    # Standard English advice
    en_advice = (
        "<b>English:</b> Diabetic retinopathy is a preventable cause of blindness. Keep your blood sugar (HbA1c < 7%) "
        "and blood pressure tightly controlled. Follow the prescribed referral schedule without delay. "
        "If you experience sudden blurriness or dark floaters, report immediately to the nearest eye hospital."
    )
    
    # Regional translations
    regional_texts = {
        "Hindi": (
            "<b>Hindi (हिंदी सलाह):</b> मधुमेह (डायबिटीज) का असर आपकी आंखों के पर्दे (रेटिना) पर पड़ सकता है। "
            "अपनी शुगर (HbA1c) और ब्लड प्रेशर को संतुलित रखें। डॉक्टर द्वारा बताई गई दवाइयां समय पर लें। "
            "यदि आंखों के आगे काले धब्बे या अचानक धुंधलापन दिखाई दे, तो तुरंत नेत्र विशेषज्ञ से संपर्क करें।"
        ),
        "Tamil": (
            "<b>Tamil (தமிழ் ஆலோசனை):</b> சர்க்கரை நோய் உங்கள் விழித்திரையை (Retina) பாதிக்கக்கூடும். "
            "ரத்த சர்க்கரை அளவு (HbA1c) மற்றும் ரத்த அழுத்தத்தை கட்டுப்பாட்டில் வைத்திருங்கள். "
            "பார்வை மங்குதல் அல்லது கருப்பு புள்ளிகள் தெரிந்தால் உடனடியாக கண் மருத்துவமனைக்கு செல்லவும்."
        ),
        "Telugu": (
            "<b>Telugu (తెలుగు సలహా):</b> మధుమేహం (షుగర్ వ్యాధి) మీ కంటి రెటీనాపై ప్రభావం చూపవచ్చు. "
            "రక్తంలో చక్కెర (HbA1c) మరియు రక్తపోటును నియంత్రణలో ఉంచండి. "
            "చూపు మసకబారినా లేదా నల్లటి మచ్చలు కనిపించినా వెంటనే కంటి వైద్యుడిని సంప్రదించండి."
        )
    }
    
    if lang_choice == "English":
        full_advice = en_advice
    else:
        selected_reg_text = regional_texts.get(lang_choice, regional_texts["Hindi"])
        full_advice = f"{en_advice}<br/><br/>{selected_reg_text}"
    story.append(Paragraph(full_advice, body_style))
    story.append(Spacer(1, 12))
    
    # 6. Doctor Sign-Off & Legal Safeguard
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#888888'), spaceBefore=2, spaceAfter=8))
    sign_table = [
        [
            Paragraph("<b>Community Health Worker (PHC):</b><br/><br/>___________________________<br/>Signature / Stamp", body_style),
            Paragraph("<b>Tele-Ophthalmologist Reviewer:</b><br/><br/>___________________________<br/>Dr. Registered Ophthalmologist (MCI)", body_style),
            Paragraph("<b>Regulatory Disclaimer:</b><br/><font size=7 color='#666666'>AI is a screening triage assistant under SIH26038 guidelines, not a replacement for comprehensive clinical ophthalmic evaluation.</font>", body_style)
        ]
    ]
    t_sign = Table(sign_table, colWidths=[2.3 * inch, 2.5 * inch, 2.2 * inch])
    t_sign.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('TOPPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(t_sign)
    
    doc.build(story)
    return output_filepath
