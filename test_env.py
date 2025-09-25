import spacy
from transformers import pipeline
import wikipediaapi

# Test Spacy
try:
    nlp = spacy.load("en_core_web_sm")
    print("Spacy loaded ✅")
except Exception as e:
    print("Spacy error:", e)

# Test HuggingFace NLI
try:
    nli = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    print("HuggingFace NLI loaded ✅")
except Exception as e:
    print("HuggingFace NLI error:", e)

# Test WikipediaAPI
try:
    wiki = wikipediaapi.Wikipedia(language="en", user_agent="test")
    print("WikipediaAPI loaded ✅")
except Exception as e:
    print("WikipediaAPI error:", e)
