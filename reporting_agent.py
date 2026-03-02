import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "mistral"

ICD10_CODES = {
    "COVID": "U07.1",
    "Lung_Opacity": "R91.8",
    "Normal": "Z00.00",
    "Viral Pneumonia": "J12.9"
}

# ==========================
# Severity Estimation
# ==========================
def estimate_severity(confidence):
    if confidence >= 0.85:
        return "Severe"
    elif confidence >= 0.60:
        return "Moderate"
    else:
        return "Mild"


# ==========================
# Confidence Calibration
# ==========================
def calibrate_confidence(confidence):
    if confidence >= 0.85:
        return "Highly Suggestive", "Low"
    elif confidence >= 0.60:
        return "Moderately Suggestive", "Moderate"
    else:
        return "Indeterminate", "High"


# ==========================
# Call Ollama API
# ==========================
def call_ollama(prompt):

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "top_p": 0.9
        }
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload)

        if response.status_code == 200:
            return response.json()["response"]
        else:
            return f"Error: Ollama returned status {response.status_code}"

    except Exception as e:
        return f"Connection Error: {str(e)}"


# ==========================
# Build Structured Prompt
# ==========================
def build_prompt(result, patient_info):

    condition = result["predicted_condition"]
    confidence_prob = result["confidence_score"]
    confidence_percent = result["confidence_percentage"]

    icd_code = ICD10_CODES.get(condition, "Not Assigned")
    severity = estimate_severity(confidence_prob)
    likelihood, uncertainty = calibrate_confidence(confidence_prob)

    prompt = f"""
You are a clinical radiology reporting assistant.

Generate a STRICTLY structured chest X-ray report.

----------------------------------------
PATIENT INFORMATION:
Name: {patient_info['name']}
Age: {patient_info['age']}
Gender: {patient_info['gender']}
Patient ID: {patient_info['patient_id']}

EXAMINATION:
Chest X-ray (AI-assisted analysis)

TECHNIQUE:
Single radiographic image analyzed using EfficientNet-B0 deep learning model.

FINDINGS:
Describe radiographic observations consistent with {condition}.

IMPRESSION:
Findings are {likelihood} of {condition}.

ICD-10 CODE:
{icd_code}

SEVERITY ASSESSMENT:
{severity}

DIAGNOSTIC CONFIDENCE:
Model confidence: {confidence_percent}%
Uncertainty level: {uncertainty}

RECOMMENDATION:
Provide appropriate clinical next steps.

DISCLAIMER:
This is AI-assisted analysis and must be verified by a licensed physician.
----------------------------------------

Use professional radiology language.
Keep report concise.
"""

    return prompt


# ==========================
# Main Function
# ==========================
def generate_medical_report(result, patient_info):

    prompt = build_prompt(result, patient_info)
    report = call_ollama(prompt)

    return report