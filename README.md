AI Fake News Detector

AI Fake News Detector is a web application that uses OCR, NLP, Machine Learning, and NLI Fact-Checking to detect whether a piece of news is real or fake. The app can handle text input, news screenshots, and supports translation from any language to English.

Features

Text Input: Paste news text directly.

OCR Support: Upload images/screenshots of news for automatic text extraction.

Translation: Automatically translates non-English news to English.

ML Prediction: Uses a pre-trained machine learning model to predict fake or real news.

NLI Fact-Checking: Uses Natural Language Inference (NLI) models to cross-check facts dynamically with live sources (Wikipedia).

Entity Extraction: Extracts important entities such as persons, organizations, positions, and locations.

Automatic NLI Server Wait: Ensures the NLI model is loaded before processing requests.

Modern UI: Clean interface with cards, AI GIFs, and responsive layout.

Screenshots


Example GIF showing AI analysis in action.

Installation

Clone the repository:

git clone https://github.com/YourUsername/ai-fake-news-detector.git
cd ai-fake-news-detector


Create a virtual environment:

python -m venv venv


Activate the virtual environment:

# PowerShell
venv\Scripts\Activate.ps1

# CMD
venv\Scripts\activate.bat


Install dependencies:

pip install -r requirements.txt


Download SpaCy English model:

python -m spacy download en_core_web_sm

Usage

Run the NLI server:

python nli_server.py


Run the Flask web app:

python app.py


Open a browser and go to:

http://127.0.0.1:5000


Paste news text or upload a screenshot to analyze.

Project Structure
fake-news/
│
├─ app.py             # Main Flask web app
├─ nli_server.py      # NLI microservice
├─ templates/
│   └─ index.html     # HTML template
├─ static/
│   ├─ style.css      # CSS styling
│   └─ ai.gif         # AI animation
├─ uploads/           # Folder for OCR uploads (ignored in Git)
├─ model.pkl          # Optional ML model
├─ vectorizer.pkl     # Optional ML vectorizer
├─ requirements.txt   # Python dependencies
└─ README.md          # Project documentation

Notes

Large ML and NLI models may take time to load initially, especially on CPU.

NLI microservice must be running before the Flask app can perform fact-checks.

For Windows users with limited C: drive space, set HuggingFace cache to another drive:

set HF_HOME=D:\huggingface_cache
set TRANSFORMERS_CACHE=D:\huggingface_cache
