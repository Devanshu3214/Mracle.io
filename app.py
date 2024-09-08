from flask import Flask, render_template, request, jsonify
import cv2
import pytesseract
import os
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import cohere  
from werkzeug.utils import secure_filename
import torch

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

app = Flask(__name__)


key1 = 'SHVrh7Hv1v7ZEEOObIn8uyJ2VAobdLU2FOOBqhHI'
cohere_api_key = key1 
co = cohere.Client(cohere_api_key)

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
model.eval()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/describe', methods=['POST'])
def describe():
    files = request.files.getlist('screenshots') 
    context = request.form.get('context') 
    image_captions = []
    extracted_texts = []

    for file in files:
        filename = secure_filename(file.filename)
        file_path = os.makedirs('uploads', exist_ok=True)
        file.save(os.path.join('uploads', filename))

        try:
            image = Image.open(os.path.join('uploads', filename)).convert("RGB")
            inputs = processor(image, return_tensors="pt")

            with torch.no_grad():
                outputs = model.generate(**inputs)
            caption = processor.decode(outputs[0], skip_special_tokens=True)
            image_captions.append(caption)

            image_cv = cv2.imread(os.path.join('uploads', filename))
            gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
            text = pytesseract.image_to_string(gray)
            extracted_texts.append(text.strip())

        except Exception as e:
            return jsonify({'error': f'Processing failed for {filename}: {str(e)}'}), 500

    combined_text = (
        f"Context: {context}\n"
        f"Image Captions: {' | '.join(image_captions)}\n"
        f"Extracted Texts: {' | '.join(extracted_texts)}"
    )

    test_case_prompt = (
        "Generate concise manual test cases based on the feature or functionality I describe, using the format shown below. For each feature, generate 5 test cases with clear and specific verification steps. The test cases should cover key scenarios including both positive and negative outcomes."
        f"{combined_text}"
    )

    try:
        response = co.generate(
            model='command-xlarge-nightly',  
            prompt=test_case_prompt,
            max_tokens=200
        )
        test_cases = response.generations[0].text.strip()
    except Exception as e:
        return jsonify({'error': f'Test case generation failed: {str(e)}'}), 500

    return test_cases

if __name__ == '__main__':
    app.run(debug=True)