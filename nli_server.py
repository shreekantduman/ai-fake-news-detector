from flask import Flask, request, jsonify
from transformers import pipeline
import os

# ------------------ SUPPRESS WARNINGS ------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = Flask(__name__)

# ------------------ LOAD LARGE NLI MODEL ONCE ------------------
print("Loading large NLI model. This may take a while...")
#nli = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
nli = pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-1")
print("NLI model loaded ✅")

@app.route('/check', methods=['POST'])
def check():
    data = request.json
    text = data.get('text', '')
    candidate_labels = data.get('candidate_labels', ['true', 'false'])
    if not text:
        return jsonify({"error": "No text provided"}), 400
    try:
        result = nli(text, candidate_labels=candidate_labels)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=5001, debug=False)
