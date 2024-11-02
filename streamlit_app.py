import openai
import os
import base64
import json
import re
from PIL import Image
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import streamlit as st
import numpy as np
from streamlit_drawable_canvas import st_canvas

import cv2
MODEL = "gpt-4o-2024-08-06"
# API_KEY = os.environ.get(
#     "OPENAI_API_KEY", "<your OpenAI API key if not set as an env var>")

API_KEY = st.secrets['OPENAI_API_KEY']
client = openai.OpenAI(api_key=API_KEY)


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Function to estimate cost based on token count
# Updated rates according to OpenAI's pricing for gpt-4o-2024-08-06
# $2.50 / 1M input tokens, $10.00 / 1M output tokens
# Assuming approximately 50% of input and output tokens are cached ($1.25 / 1M cached tokens)


def estimate_cost(token_count, input_token_rate=2.50, output_token_rate=10.00, cached_rate=1.25):
    input_cost = (token_count / 1_000_000) * input_token_rate
    output_cost = (token_count / 1_000_000) * output_token_rate
    # Assuming roughly half of input tokens are cached
    cached_input_cost = (token_count / 1_000_000) * cached_rate
    # Assuming roughly half of output tokens are cached
    cached_output_cost = (token_count / 1_000_000) * cached_rate
    return (input_cost + output_cost) * 0.5 + (cached_input_cost + cached_output_cost) * 0.5


class DoctorDetails(BaseModel):
    name: Optional[str] = None
    qualifications: Optional[str] = None
    specialization: Optional[str] = None
    registration_number: Optional[str] = None


class PatientDetails(BaseModel):
    name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    date: Optional[str] = None


class PrescriptionItem(BaseModel):
    medicine_name: Optional[str] = None
    dosage: Optional[str] = None
    quantity: Optional[int] = None


class MedicalPrescriptionExtraction(BaseModel):
    hospital_name: Optional[str] = None
    doctor_details: Optional[DoctorDetails] = None
    patient_details: Optional[PatientDetails] = None
    diagnosis: Optional[str] = None
    prescription: Optional[List[PrescriptionItem]] = None


st.title("Medical Prescription Data Extraction.")

canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",
    stroke_width=2,
    stroke_color="#000000",
    background_color="#eee",
    height=400,
    width=600,
    drawing_mode="freedraw",
    key="canvas"
)

if st.button("Submit"):
    if canvas_result.image_data is not None:
        img = canvas_result.image_data
        img = np.array(img, dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        pil_img = Image.fromarray(img)

        pil_img.save("temp_image.png")
        base64_image = encode_image("temp_image.png")
        token_count = len(base64_image) // 4

        # estimated_cost = estimate_cost(token_count)
        # st.write(f"Estimated Cost for Processing: {estimated_cost:.4f}")

        response = client.chat.completions.create(
            model=MODEL,  # Use the latest GPT-4o model identifier
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "You are an AI assistant that extracts data from medical prescriptions. Please strictly return the information in the following JSON format, without any extra text or explanation:\n"
                            "{\n"
                            "  \"hospital_name\": \"<Hospital Name>\",\n"
                            "  \"patient_details\": {\n"
                            "    \"name\": \"<Patient Name>\",\n"
                            "    \"date\": \"<Date>\"\n"
                            "  },\n"
                            "  \"diagnosis\": \"<Diagnosis>\",\n"
                            "  \"prescription\": [\n"
                            "    {\n"
                            "      \"medicine_name\": \"<Medicine Name>\",\n"
                            "      \"dosage\": \"<Dosage>\",\n"
                            "      \"quantity\": <Quantity>\n"
                            "      \"when_to_use\": [<When to Use>]\n"
                            "    }\n"
                            "  ]\n"
                            "}"
                         "Note: Dosage patterns may be represented as follows: \"1-1-0\" means morning and afternoon; "
                         "\"1-0-1\" means morning and evening; \"1-1-1\" means morning, afternoon, and evening.\n"
                         "For the \"when_to_use\" field, please return an array such as [\"morning\", \"afternoon\", \"evening\"] based on the dosage pattern.\n"
                         "Before proceeding, please verify the extracted data twice to ensure accuracy, including correcting any spelling, "
                         "grammar, or formatting errors that might be present."
                         },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                                "detail": "high"  # Adjust to "high" for detailed analysis
                            }
                        }
                    ]
                }
            ],
            max_tokens=300,
        )

        json_response = response.choices[0].message.content.strip()
        json_response = re.sub(r'```(?:json)?', '', json_response).strip()
        try:
            json_data = json.loads(json_response)
            st.subheader("Extracted Data in JSON Format:")
            st.json(json_data)
            with open("output.json", "w") as json_file:
                json.dump(json_data, json_file, indent=4)
        except json.JSONDecodeError as e:
            st.error(
                "Failed to parse JSON response. Here is the raw response instead:")
            st.text(json_response)
