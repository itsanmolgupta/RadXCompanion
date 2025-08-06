import streamlit as st
from PIL import Image
from transformers import AutoTokenizer, VisionEncoderDecoderModel, ViTFeatureExtractor, BartTokenizer, BartForConditionalGeneration
import torch
import tempfile
import os
import re
import google.generativeai as genai
from datetime import datetime
from fpdf import FPDF

# Load Findings Generation Model (ViT-GPT2)
findings_model_dir = "findings-model-vitgpt2"
findings_tokenizer = AutoTokenizer.from_pretrained(findings_model_dir)
findings_model = VisionEncoderDecoderModel.from_pretrained(findings_model_dir)
feature_extractor = ViTFeatureExtractor.from_pretrained(findings_model_dir)

# Load the BART model and tokenizer for impression generation
impression_model_dir = "impression-model-bart"
bart_tokenizer = BartTokenizer.from_pretrained(impression_model_dir)
bart_model = BartForConditionalGeneration.from_pretrained(impression_model_dir)

# Configure Gemini API
API_KEY = ""
genai.configure(api_key=API_KEY)

# Define use_gemini variable
verify_image = True
use_gemini = True 

# Function to check if the uploaded image is a frontal chest X-ray
def check_frontal_xray(image):
    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content(["Is this a frontal chest X-ray? Reply 'Yes' or 'No' only.", image])
    return response.text.strip().lower() == "yes"

# Function to generate findings
def generate_findings(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
    with torch.no_grad():
        gen_kwargs = {"max_length": 512, "num_beams": 4}
        generated_ids = findings_model.generate(pixel_values, **gen_kwargs)
    findings = findings_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return findings

# Function to generate impression
def generate_impression(findings):
    inputs = bart_tokenizer(findings, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        summary_ids = bart_model.generate(**inputs, max_length=150, num_beams=4, early_stopping=True)
    impression = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return impression

# Function to enhance the report using Gemini (unchanged)
def analyze_with_gemini(findings, impression):
    if not use_gemini:
        return findings, impression
    
    model = genai.GenerativeModel("gemini-1.5-pro")
    prompt = f"""
    You are a qualified radiologist. Based on following chest X-ray report:

    **Findings:** {findings}
    **Impression:** {impression}

    1. Highlight sentences indicating abnormalities by wrapping them in **<span style='color:red'>** and **</span>**.
    2. Return the highlighted report in the following format:
       **Findings:** [highlighted findings text]
       **Impression:** [highlighted impression text]
    """
    response = model.generate_content([prompt])
    enhanced_text = response.text.strip()
    
    findings_section = enhanced_text.split("**Findings:**")[1].split("**Impression:**")[0].strip()
    impression_section = enhanced_text.split("**Impression:**")[1].strip()
    
    return findings_section, impression_section

# Function to generate PDF report (corrected)
def generate_pdf_report(patient_name, age, sex, address, findings_text, impression_text, uploaded_file):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=25)
    pdf.add_page()

    # Header
    pdf.set_font("Arial", style="B", size=18)
    pdf.cell(0, 10, "Chest X-ray Report", ln=1, align="C")
    pdf.ln(5)

    # Patient Details
    pdf.set_font("Arial", "B", 14)
    pdf.cell(100, 10, "Patient Details:", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.cell(100, 10, f"Name: {patient_name}", ln=True)
    pdf.cell(100, 10, f"Address: {address}", ln=True)
    pdf.cell(100, 10, f"Age: {age} Yrs", ln=True)
    pdf.cell(100, 10, f"Sex: {sex}", ln=True)
    
    pdf.set_font("Arial", "B", 12)
    pdf.cell(100, 10, "Report Generated on:", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.cell(100, 10, datetime.now().strftime("%d %B, %Y\n%I:%M %p"), ln=True)

    pdf.ln(5)

    # Report Content
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Generated Report:", ln=1, align="L")

    # Findings
    pdf.set_font("Arial", "BU", 12)
    x_before = pdf.get_x()
    y_before = pdf.get_y()
    pdf.cell(100, 10, "Findings", ln=0, align="L")
    pdf.set_xy(x_before + 120, y_before)
    pdf.cell(70, 10, "X-ray Image", ln=1, align="C")

    # Standardize sentence separators for findings
    findings_text = findings_text.replace('.- ', '. ')
    
    # Split findings into normal and abnormal parts
    split_parts = re.split(r"(<span style='color:red'>.*?</span>)", findings_text, flags=re.DOTALL)
    
    # Extract all sentences with abnormality status
    findings_sentences = []
    for part in split_parts:
        if part.startswith("<span style='color:red'>"):
            # Abnormal part: remove tags and split into sentences
            content = part.replace("<span style='color:red'>", "").replace("</span>", "")
            sentences = [s.strip() for s in content.split('. ') if s.strip()]
            for s in sentences:
                if not s.endswith('.'):
                    s += '.'
                findings_sentences.append((s, True))  # True indicates abnormal
        elif part.strip():
            # Normal part: split into sentences
            sentences = [s.strip() for s in part.split('. ') if s.strip()]
            for s in sentences:
                if not s.endswith('.'):
                    s += '.'
                findings_sentences.append((s, False))  # False indicates normal

    # Generate Findings in PDF
    pdf.set_font("Arial", size=10)
    x_before = pdf.get_x()
    col_width = 115
    for sentence, is_abnormal in findings_sentences:
        if is_abnormal:
            pdf.set_font("Arial", "B", 10)  # Bold for abnormalities
        else:
            pdf.set_font("Arial", "", 10)   # Regular for normal
        pdf.multi_cell(col_width, 8, f"- {sentence}", align="L")
        pdf.set_x(x_before)

    # Display uploaded image
    if uploaded_file:
        y_after_findings = pdf.get_y()
        pdf.set_xy(x_before + 120, y_before)
        uploaded_file.seek(0)
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name
        pdf.image(temp_file_path, w=70)
        os.remove(temp_file_path)
        pdf.set_xy(x_before, max(y_after_findings, y_before + 60))

    pdf.ln(5)

    # Impression
    pdf.set_font("Arial", "BU", 12)
    pdf.cell(0, 10, "Impression", ln=1, align="L")

    # Standardize sentence separators for impression
    impression_text = impression_text.replace('.- ', '. ')
    
    # Split impression into normal and abnormal parts
    split_parts_impression = re.split(r"(<span style='color:red'>.*?</span>)", impression_text, flags=re.DOTALL)
    
    # Extract all sentences with abnormality status
    impression_sentences = []
    for part in split_parts_impression:
        if part.startswith("<span style='color:red'>"):
            # Abnormal part: remove tags and split into sentences
            content = part.replace("<span style='color:red'>", "").replace("</span>", "")
            sentences = [s.strip() for s in content.split('. ') if s.strip()]
            for s in sentences:
                if not s.endswith('.'):
                    s += '.'
                impression_sentences.append((s, True))  # True indicates abnormal
        elif part.strip():
            # Normal part: split into sentences
            sentences = [s.strip() for s in part.split('. ') if s.strip()]
            for s in sentences:
                if not s.endswith('.'):
                    s += '.'
                impression_sentences.append((s, False))  # False indicates normal

    # Generate Impression in PDF
    pdf.set_font("Arial", size=10)
    for sentence, is_abnormal in impression_sentences:
        if is_abnormal:
            pdf.set_font("Arial", "B", 10)  # Bold for abnormalities
        else:
            pdf.set_font("Arial", "", 10)   # Regular for normal
        pdf.multi_cell(0, 6, f"- {sentence}", align="L")

    pdf.ln(15)
    pdf.set_font("Arial", "I", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.multi_cell(0, 5, "Disclaimer: This is an AI-generated report for informational purposes only and should not be considered a definitive diagnosis. Always consult a qualified medical professional.")
    
    return pdf.output(dest="S").encode("latin1")

# Streamlit UI
def report_generation_page():
    st.title("Chest X-ray Report Generation")
    st.write("Generate Findings and Impression for a Chest X-ray Image.")

    if "last_uploaded_file" not in st.session_state:
        st.session_state.last_uploaded_file = None
    if "findings" not in st.session_state:
        st.session_state.findings = ""
    if "impression" not in st.session_state:
        st.session_state.impression = ""

    uploaded_file = st.file_uploader("Upload a Chest X-ray Image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_id = uploaded_file.file_id if hasattr(uploaded_file, 'file_id') else uploaded_file.name
        is_new_upload = st.session_state.last_uploaded_file != file_id

        col1, col2 = st.columns([1, 2])
        
        with col1:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Chest X-ray', use_container_width=True)
        
        with col2:
            if verify_image:
                with st.spinner('Verifying Image...'):
                    if not check_frontal_xray(image):
                        st.warning("Uploaded image is not a frontal chest X-ray. Please upload a valid image.")
                        return
                
            if is_new_upload:
                with st.spinner('Generating Findings...'):
                    findings = generate_findings(image)
                with st.spinner('Generating Impression...'):
                    impression = generate_impression(findings)
                
                if use_gemini:
                    with st.spinner('Highlighting Abnormalities...'):
                        findings, impression = analyze_with_gemini(findings, impression)
                
                st.session_state.findings = findings
                st.session_state.impression = impression
                st.session_state.last_uploaded_file = file_id
            
            # Display findings and impression with HTML formatting
            st.write("**Generated Findings:**")
            st.markdown(st.session_state.findings, unsafe_allow_html=True)
            st.write("**Generated Impression:**")
            st.markdown(st.session_state.impression, unsafe_allow_html=True)
        
        st.markdown("---")
        
        with st.form(key="report_form"):
            st.header("Edit and Finalize Report")
            # Strip HTML tags for editing and comparison
            findings_plain = st.session_state.findings.replace("<span style='color:red'>", "").replace("</span>", "")
            impression_plain = st.session_state.impression.replace("<span style='color:red'>", "").replace("</span>", "")
            edited_findings = st.text_area("Edit Findings", value=findings_plain, height=200)
            edited_impression = st.text_area("Edit Impression", value=impression_plain, height=100)
            
            st.markdown("### Enter Patient Details")
            patient_name = st.text_input("Patient Name")
            age = st.text_input("Age")
            sex = st.selectbox("Sex", ["Male", "Female", "Other"])
            address = st.text_input("Address")
            
            submit_button = st.form_submit_button("Generate PDF Report")
        
        if submit_button:
            # Check if all required fields are filled
            if not patient_name:
                st.error("Please enter the patient's name.")
            elif not age:
                st.error("Please enter the patient's age.")
            elif not address:
                st.error("Please enter the patient's address.")
            else:
                # Determine if edited text differs from generated text (without HTML tags)
                final_findings = st.session_state.findings
                final_impression = st.session_state.impression
                
                if edited_findings != findings_plain or edited_impression != impression_plain:
                    final_findings, final_impression = analyze_with_gemini(edited_findings, edited_impression)
                
                pdf_bytes = generate_pdf_report(patient_name, age, sex, address, final_findings, final_impression, uploaded_file)
                st.download_button(
                    label="Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"{patient_name.replace(' ', '_')}_Chest_X-ray_Report.pdf",
                    mime="application/pdf"
                )