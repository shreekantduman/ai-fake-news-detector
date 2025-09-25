import os
import re
import time
import pickle
import requests
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from PIL import Image
import pytesseract
from deep_translator import GoogleTranslator
import spacy
import wikipediaapi

# -------------------- CONFIG --------------------
UPLOAD_FOLDER = 'uploads'
MODEL_PATH = 'model.pkl'
VECT_PATH = 'vectorizer.pkl'
NLI_SERVER_URL = "http://localhost:5001/check"
MAX_WAIT = 120  # seconds

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -------------------- LOAD NLP & WIKIPEDIA --------------------
nlp = spacy.load("en_core_web_sm")
wiki_wiki = wikipediaapi.Wikipedia(
    language="en",
    user_agent="AI-Fake-News-Detector/1.0"
)

# -------------------- LOAD ML MODEL (OPTIONAL) --------------------
model = None
vectorizer = None
if os.path.exists(MODEL_PATH) and os.path.exists(VECT_PATH):
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(VECT_PATH, 'rb') as f:
        vectorizer = pickle.load(f)

# -------------------- HELPER FUNCTIONS --------------------
def wait_for_nli_server(timeout=MAX_WAIT, interval=5):
    """Wait until the NLI microservice is ready."""
    start_time = time.time()
    while True:
        try:
            resp = requests.post(NLI_SERVER_URL, json={"text": "Test", "candidate_labels": ["true","false"]}, timeout=5)
            if resp.status_code == 200:
                print("✅ NLI server is ready.")
                return True
        except requests.exceptions.RequestException:
            pass

        if time.time() - start_time > timeout:
            print("⚠️ NLI server did not respond within timeout. ML-only fallback will be used.")
            return False

        print("Waiting for NLI server to be ready...")
        time.sleep(interval)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = ' '.join(text.split())
    return text

def translate_to_en(text):
    try:
        return GoogleTranslator(source='auto', target='en').translate(text)
    except:
        return text

def extract_entities(text):
    doc = nlp(text)
    entities = {"PERSON": [], "ORG": [], "GPE": [], "POSITION": []}
    for ent in doc.ents:
        if ent.label_ in entities:
            entities[ent.label_].append(ent.text)
    positions = ["Prime Minister", "Chief Minister", "President", "Governor", "Minister"]
    for pos in positions:
        if pos.lower() in text.lower():
            entities["POSITION"].append(pos)
    return entities

def fetch_wiki_summary(query):
    try:
        page = wiki_wiki.page(query)
        if page.exists():
            return page.summary
    except:
        pass
    return ""

def fact_check_nli(text):
    entities = extract_entities(text)
    candidates = []
    for ent_type in ["PERSON", "ORG", "GPE", "POSITION"]:
        for ent in entities[ent_type]:
            summary = fetch_wiki_summary(ent)
            if summary:
                candidates.append(summary)
    if not candidates:
        summary = fetch_wiki_summary(text)
        if summary:
            candidates.append(summary)

    # Call NLI server
    for premise in candidates:
        try:
            resp = requests.post(NLI_SERVER_URL, json={"text": text, "candidate_labels": [premise]}, timeout=10)
            data = resp.json()
            labels = data.get('labels', [])
            if labels:
                label = labels[0].lower()
                if 'contradiction' in label:
                    return False
                elif 'entailment' in label:
                    return True
        except:
            continue
    return None

def ml_predict(text):
    if vectorizer is None or model is None:
        return None
    try:
        vec = vectorizer.transform([clean_text(text)])
        pred = model.predict(vec)[0]
        return 'Fake' if int(pred) == 1 else 'Real'
    except:
        return None

# -------------------- WAIT FOR NLI SERVER --------------------
wait_for_nli_server()

# -------------------- FLASK ROUTE --------------------
@app.route('/', methods=['GET', 'POST'])
def home():
    warning, extracted, translated, ml_result, final_result, entities = None, None, None, None, None, {}

    if request.method == 'POST':
        text_input = request.form.get('news_text', '').strip()
        file = request.files.get('news_image')

        if not text_input and (not file or file.filename == ''):
            warning = "Provide text or upload image."
            return render_template('index.html', warning=warning)

        if file and file.filename:
            filename = secure_filename(file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)
            try:
                extracted = pytesseract.image_to_string(Image.open(path))
            except:
                extracted = ''
            text_input = extracted or text_input

        translated = translate_to_en(text_input)
        ml_result = ml_predict(translated)
        entities = extract_entities(translated)

        fact_result = fact_check_nli(translated)
        if fact_result is True:
            final_result = "Real"
        elif fact_result is False:
            final_result = "Likely Fake"
        else:
            final_result = ml_result or "Unknown"

    return render_template('index.html',
                           warning=warning,
                           extracted=extracted,
                           translated=translated,
                           ml_result=ml_result,
                           final_result=final_result,
                           entities=entities)

# -------------------- RUN SERVER --------------------
if __name__ == '__main__':
    app.run(debug=True)
