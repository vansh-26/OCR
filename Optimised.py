import cv2
import easyocr
import re
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins="*",  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # You can specify the HTTP methods you want to allow
    allow_headers=["*"],  # You can specify the HTTP headers you want to allow
)

# Function to extract text from an image using EasyOCR
def extract_text_from_image(contents):
    reader = easyocr.Reader(['en'])
    result = reader.readtext(contents)
    text = ' '.join([entry[1] for entry in result])
    return text

# Function to extract name using OCR Reader and specific parameters
async def name_search(reader, contents, counter_condition, height_condition, width_condition, y_condition):
    np_array = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    # Preprocessing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 10))
    dilate = cv2.dilate(thresh, kernel, iterations=1)

    # Find contours
    cnts, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by x-coordinate
    cnts = sorted(cnts, key=lambda x: cv2.boundingRect(x)[1])

    # Start of name search
    roi_counter = 0
    name = ""
    screen_height, screen_width = image.shape[:2]

    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if (h / screen_height) * 100 >= height_condition[0] and (h / screen_height) * 100 < height_condition[1] \
                and (w / screen_width) * 100 >= width_condition and (y / screen_height) * 100 > y_condition:
            roi_counter += 1
            roi = gray[y:y + h, x:x + w]
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if roi_counter == counter_condition:
                name = reader.readtext(roi, detail=0, paragraph=False)[0]
                break

    # End of name search
    return name

def extract_info_aadhaar(text):
    dob_pattern = r"DOB (\d{2}/\d{2}/\d{4})"
    aadhar_pattern = r"\b\d{4}\s?\d{4}\s?\d{4}\b"

    dob_match = re.search(dob_pattern, text)
    aadhar_match = re.search(aadhar_pattern, text)

    dob = dob_match.group(1) if dob_match else None
    aadhar = aadhar_match.group() if aadhar_match else None

    return dob, aadhar

def extract_info_pan(text):
    name_pattern = r'\bName(?:\s*:\s*|\s+)(\w+\s+\w+)\b'
    name_match = re.search(name_pattern, text)
    name = name_match.group(1) if name_match else None

    dob_pattern = r'\b\d{2}/\d{2}/\d{4}\b'
    pan_pattern = r"[A-Z]{5}[0-9]{4}[A-Z]{1}"

    dob_match = re.search(dob_pattern, text)
    pan_match = re.search(pan_pattern, text)

    dob = dob_match.group() if dob_match else None
    pan = pan_match.group() if pan_match else None

    return name, dob, pan

async def find_id_type(extracted_text):
    if "Government of India" in extracted_text:
        return "aadhaar"
    elif "INCOME TAX DEPARTMENT" in extracted_text:
        return "PAN"

@app.post("/process-image")
async def process_image(file: UploadFile = File(...)):

    contents = await file.read()
    # Use EasyOCR to extract text from the image
    extracted_text = extract_text_from_image(contents)

    id_type = await find_id_type(extracted_text)
    reader = easyocr.Reader(['en'])

    if id_type == "aadhaar":
        # Extract name, DOB, and Aadhar number from the text
        dob, pan = extract_info_aadhaar(extracted_text)

        name = await name_search(reader, contents, 4, (2, 10), 6, 0)

    elif id_type == "PAN":
        # Extract name, DOB, and Aadhar number from the text
        name, dob, pan = extract_info_pan(extracted_text)
        if name is None:
            name = await name_search(reader, contents, 1, (2, 10), 6, 23)

    # Return the results
    return {"idtype": id_type, "filename": file.filename, "text": extracted_text, "name": name, "dob": dob, "Identity Number": pan}
