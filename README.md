# Fake News Detector — Full Enhanced Package

Features:
- Improved ML model (LinearSVC, TF-IDF with ngrams, balanced)
- OCR support (upload screenshots) using Tesseract
- Translation using deep-translator (GoogleTranslator)
- Authenticity scanner: Google News RSS + optional NewsAPI integration (set NEWSAPI_KEY env var)

Setup:
1. Create venv and install requirements:
   python -m venv venv
   source venv/bin/activate   # or venv\Scripts\activate on Windows
   pip install -r requirements.txt
2. Place your BharatFakeNewsKosh.xlsx in project root
3. Train model:
   python train_model.py --input BharatFakeNewsKosh.xlsx --out-model model.pkl --out-vectorizer vectorizer.pkl
4. (Optional) Set NEWSAPI_KEY environment variable to enable NewsAPI results
5. Run the app:
   python app.py
6. Open http://127.0.0.1:5000

Notes:
- Tesseract OCR: you must install the Tesseract binary on your system for pytesseract to work.
  - Windows: install from https://github.com/tesseract-ocr/tesseract
  - Ubuntu: sudo apt-get install tesseract-ocr
