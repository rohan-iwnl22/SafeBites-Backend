import cv2
import easyocr
import pandas as pd
import numpy as np
import re
import os
from flask import Flask, request, jsonify
from fuzzywuzzy import process
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load and preprocess additives dataset
ADDATIVES_CSV = "LAST_FINAL_1.csv"  # Replace with actual path
ADDATIVES_DF, ADDITIVES_DICT = None, None
ENHANCED_DICT = {}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def clean_dataset(filepath):
    df = pd.read_csv(filepath)
    if len(df.columns) == 2:
        df.columns = ['ADDITIVES', 'MAX_LEVEL']
    df = df.dropna(how='all').reset_index(drop=True)
    df['ADDITIVES'] = df['ADDITIVES'].fillna('').str.upper()
    df['MAX_LEVEL'] = df['MAX_LEVEL'].fillna('GMP').replace('', 'GMP')
    df = df.drop_duplicates()
    
    return df, dict(zip(df['ADDITIVES'], df['MAX_LEVEL']))

def preprocess_additives(additives_df):
    additives_dict = {}
    for _, row in additives_df.iterrows():
        additive_name = row['ADDITIVES'].strip().upper()
        additives_dict[additive_name] = row['MAX_LEVEL']

        if '"' in additive_name:
            additives_dict[additive_name.replace('"', '')] = row['MAX_LEVEL']
        if '(' in additive_name:
            additives_dict[re.sub(r'\s*\([^)]*\)', '', additive_name).strip()] = row['MAX_LEVEL']
        if ',' in additive_name:
            parts = [p.strip() for p in additive_name.split(',')]
            if len(parts) > 1 and len(parts[0]) > 3:
                additives_dict[parts[0]] = row['MAX_LEVEL']
    return additives_dict

def correct_text(text, additives_dict):
    words = text.upper().split()
    corrected_words = []
    
    for word in words:
        if word in additives_dict:
            corrected_words.append(word)
            continue

        match, score = process.extractOne(word, additives_dict.keys())
        if score > 90:
            corrected_words.append(match)
        else:
            corrected_words.append(word)
    
    return ' '.join(corrected_words)

def detect_additives(text, additives_dict):
    text = text.upper()
    detected_additives = []
    
    for additive in additives_dict.keys():
        if additive in text:
            match = re.search(rf"{additive}\s*(\d+\.?\d*)", text)
            detected_level = match.group(1) if match else "Not found"

            detected_additives.append({
                'additive': additive,
                'detected_level': detected_level,
                'max_level': additives_dict[additive]
            })
    
    return detected_additives

def analyze_additives(detected_list):
    if not detected_list:
        return {'count': 0, 'gmp_count': 0, 'numeric_levels': [], 'categories': {}}

    gmp_count = sum(1 for item in detected_list if item['max_level'] == 'GMP')
    
    numeric_levels = []
    for item in detected_list:
        if item['detected_level'] != "Not found":
            try:
                numeric_levels.append(float(item['detected_level']))
            except ValueError:
                pass

    return {
        'count': len(detected_list),
        'gmp_count': gmp_count,
        'numeric_levels': numeric_levels,
        'avg_level': np.mean(numeric_levels) if numeric_levels else 0,
        'max_level': max(numeric_levels) if numeric_levels else 0,
        'detailed_levels': detected_list
    }

@app.route('/extract', methods=['POST'])
@cross_origin()
def extract_text():
    print("Request received at /extract")
    print("Files in request:", list(request.files.keys()))
    print("Form data keys:", list(request.form.keys()))
    print("Content type:", request.content_type)
    # Check if the post request has the file part
    if 'image' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['image']
    
    # If user does not select file, browser might submit an empty part without filename
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and allowed_file(file.filename):
        # Ensure upload directory exists
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        # Secure the filename and save temporarily
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        try:
            # Process the image
            reader = easyocr.Reader(['en'])
            results = reader.readtext(file_path, detail=0)
            extracted_text = ' '.join(results)
            
            print(f"Extracted Text: {extracted_text}")
            
            corrected_text = correct_text(extracted_text, ENHANCED_DICT)
            print(f"Corrected Text: {corrected_text}")

            # Perform analysis
            detected = detect_additives(corrected_text, ENHANCED_DICT)
            analysis = analyze_additives(detected)
            
            response = {
                "detected_additives": detected,
                "analysis": analysis,
                "extracted_text": extracted_text,
                "corrected_text": corrected_text
            }
            
            return jsonify(response)
        except Exception as e:
            return jsonify({"error": f"Error processing image: {str(e)}"}), 500
        finally:
            # Clean up - remove the uploaded file
            if os.path.exists(file_path):
                os.remove(file_path)
    else:
        return jsonify({"error": "Invalid file type"}), 400

@app.route('/analyze', methods=['POST'])
@cross_origin()
def analyze_text():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
        
    label_text = request.json.get("text")
    if not label_text:
        return jsonify({"error": "No text provided"}), 400

    detected = detect_additives(label_text, ENHANCED_DICT)
    analysis = analyze_additives(detected)

    return jsonify({
        "detected_additives": detected,
        "analysis": analysis
    })

if __name__ == '__main__':
    # Initialize data and create upload directory
    ADDATIVES_DF, ADDITIVES_DICT = clean_dataset(ADDATIVES_CSV)
    ENHANCED_DICT = preprocess_additives(ADDATIVES_DF)
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    
    app.run(debug=True, host='0.0.0.0', port=5000)