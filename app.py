import cv2
import easyocr
import pandas as pd
import numpy as np
import re
from flask import Flask, request, jsonify
from fuzzywuzzy import process
from collections import defaultdict

app = Flask(__name__)

# Load and preprocess additives dataset
ADDATIVES_CSV = "LAST_FINAL_1.csv"  # Replace with actual path
ADDATIVES_DF, ADDITIVES_DICT = None, None
ENHANCED_DICT = {}

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
        match, score = process.extractOne(word, additives_dict.keys())
        if score > 80:  # Confidence threshold
            corrected_words.append(match)
        else:
            corrected_words.append(word)
    
    return ' '.join(corrected_words)

def detect_additives(text, additives_dict):
    text = text.upper()
    detected_additives = []
    
    for additive in additives_dict.keys():
        if additive in text:
            detected_additives.append({'additive': additive, 'max_level': additives_dict[additive]})
    
    return detected_additives

def analyze_additives(detected_list):
    if not detected_list:
        return {'count': 0, 'gmp_count': 0, 'numeric_levels': [], 'categories': {}}

    gmp_count = sum(1 for item in detected_list if item['max_level'] == 'GMP')
    numeric_levels = [float(re.search(r'(\d+(?:\.\d+)?)', str(item['max_level'])).group(1)) for item in detected_list if item['max_level'] != 'GMP' and re.search(r'\d+', str(item['max_level']))]

    return {
        'count': len(detected_list),
        'gmp_count': gmp_count,
        'numeric_levels': numeric_levels,
        'avg_level': np.mean(numeric_levels) if numeric_levels else 0,
        'max_level': max(numeric_levels) if numeric_levels else 0
    }

@app.route('/extract', methods=['POST'])
def extract_text():
    print("Request files:", request.files)
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No image provided"}), 400

    file_path = "temp.jpg"
    file.save(file_path)

    reader = easyocr.Reader(['en'])
    results = reader.readtext(file_path, detail=0)
    extracted_text = ' '.join(results)

    corrected_text = correct_text(extracted_text, ENHANCED_DICT)

    # Auto-send to analysis
    response = analyze_text(corrected_text)

    return jsonify(response)

@app.route('/analyze', methods=['POST'])
def analyze_text(text=None):
    label_text = text if text else request.json.get("text")
    if not label_text:
        return {"error": "No text provided"}  # ✅ Return a dictionary

    detected = detect_additives(label_text, ENHANCED_DICT)
    analysis = analyze_additives(detected)

    return {"detected_additives": detected, "analysis": analysis}

if __name__ == '__main__':
    ADDATIVES_DF, ADDITIVES_DICT = clean_dataset(ADDATIVES_CSV)
    ENHANCED_DICT = preprocess_additives(ADDATIVES_DF)
    
    app.run(debug=True)
