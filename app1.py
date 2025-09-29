"""
app.py - Fake News Detector (OCR + Translation + ML + NewsAPI + Wikipedia cross-check)

Author: Shreekant Suman
"""

import os
import re
import requests
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from PIL import Image
import pytesseract
from deep_translator import GoogleTranslator
import spacy
import wikipediaapi
from bs4 import BeautifulSoup
import pickle

# ---------------------- CONFIG ----------------------
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "").strip()
MODEL_PATH = "model.pkl"
VECT_PATH = "vectorizer.pkl"

# Wikipedia API
WIKI = wikipediaapi.Wikipedia(
    language="en",
    user_agent="AI-Fake-News-Detector/1.0 (contact: your-email@example.com)"
)

# SpaCy NLP
try:
    nlp = spacy.load("en_core_web_sm")
except:
    raise RuntimeError("Run: python -m spacy download en_core_web_sm")

# Load ML model if available
model, vectorizer = None, None
if os.path.exists(MODEL_PATH) and os.path.exists(VECT_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(VECT_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    print("✅ ML model loaded")
else:
    print("⚠️ No ML model found. Only NewsAPI + Wikipedia will be used.")

# ---------------------- HELPERS ----------------------

def ocr_from_file(path):
    img = Image.open(path).convert("RGB")
    return pytesseract.image_to_string(img)

def translate_to_en(text):
    try:
        return GoogleTranslator(source="auto", target="en").translate(text)
    except:
        return text

def clean_text(text):
    t = text.lower()
    t = re.sub(r"http\S+", " ", t)
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

def extract_entities(text):
    doc = nlp(text)
    entities = {"PERSON": [], "ORG": [], "GPE": [], "POSITION": []}
    for ent in doc.ents:
        if ent.label_ in entities:
            entities[ent.label_].append(ent.text)
    for pos in ["Prime Minister", "President", "Chief Minister", "Governor", "Minister"]:
        if pos.lower() in text.lower():
            entities["POSITION"].append(pos)
    return entities

def fetch_wiki_summary(topic):
    try:
        page = WIKI.page(topic)
        if page.exists():
            return page.summary
    except:
        pass
    return ""

def scrape_url(url):
    headers = {"User-Agent": "Mozilla/5.0 FakeNewsDetector/1.0"}
    try:
        r = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")

        title = soup.find("meta", property="og:title")
        title = title["content"] if title else (soup.title.string if soup.title else "")

        article = soup.find("article")
        if article:
            body = " ".join(p.get_text() for p in article.find_all("p"))
        else:
            body = " ".join(p.get_text() for p in soup.find_all("p"))
        return title.strip(), body.strip()
    except:
        return "", ""

def verify_with_newsapi(query):
    if not NEWSAPI_KEY:
        return False, None
    try:
        url = "https://newsapi.org/v2/everything"
        params = {"q": query, "language": "en", "pageSize": 5}
        resp = requests.get(url, params=params, headers={"X-Api-Key": NEWSAPI_KEY})
        data = resp.json()
        if data.get("status") == "ok" and data.get("totalResults", 0) > 0:
            article = data["articles"][0]
            return True, article.get("source", {}).get("name")
    except:
        pass
    return False, None

def strict_wiki_check(claim, entities):
    """Detect contradictions in claims vs Wikipedia."""
    for person in entities.get("PERSON", []):
        summary = fetch_wiki_summary(person)
        if summary:
            if "sri lanka" in claim.lower() and "sri lanka" not in summary.lower():
                return False
            if "india" in summary.lower() and "sri lanka" in claim.lower():
                return False

    # Generic science myths
    if "moon is hollow" in claim.lower():
        return False
    if "earth is flat" in claim.lower():
        return False
    if "sun rises in the west" in claim.lower():
        return False

    return True

def ml_predict(text):
    if not model or not vectorizer:
        return None
    vec = vectorizer.transform([clean_text(text)])
    return "Fake" if int(model.predict(vec)[0]) == 1 else "Real"

# ---------------------- FLASK APP ----------------------
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def home():
    warning, extracted, translated, ml_result, final_result = None, None, None, None, None
    entities, source_verified = {}, None

    if request.method == "POST":
        user_text = (request.form.get("news_text") or "").strip()
        user_url = (request.form.get("news_url") or "").strip()
        file = request.files.get("news_image")

        # URL scraping
        if user_url:
            title, body = scrape_url(user_url)
            user_text = f"{title} {body}".strip()

        # OCR
        if file and file.filename:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(file.filename))
            file.save(filepath)
            try:
                extracted = ocr_from_file(filepath)
                if extracted.strip():
                    user_text = extracted
            except Exception as e:
                warning = f"OCR failed: {e}"

        if not user_text:
            warning = "No text found to analyze."
            return render_template("index.html", warning=warning)

        translated = translate_to_en(user_text)
        entities = extract_entities(translated)

        # NewsAPI check
        verified, source = verify_with_newsapi(" ".join(translated.split()[:10]))
        if verified:
            consistent = strict_wiki_check(translated, entities)
            if not consistent:
                final_result = "Likely Fake (entity-role contradiction)"
            else:
                final_result = f"Real (verified via {source})"
            ml_result = ml_predict(translated)
        else:
            # No NewsAPI support → rely on Wiki + ML
            consistent = strict_wiki_check(translated, entities)
            if consistent:
                final_result = "Real (knowledge supported)"
            else:
                final_result = "Fake (contradicted by knowledge)"
            ml_result = ml_predict(translated)

    return render_template("index.html",
                           warning=warning,
                           extracted=extracted,
                           translated=translated,
                           ml_result=ml_result,
                           final_result=final_result,
                           entities=entities)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
