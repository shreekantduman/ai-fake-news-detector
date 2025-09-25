"""Train improved TF-IDF + LinearSVC classifier for Fake News detection.
Usage:
  python train_model.py --input BharatFakeNewsKosh.xlsx --out-model model.pkl --out-vectorizer vectorizer.pkl
"""
import argparse, re, string, pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def clean_text(text):
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'http\S+', '', text)   # remove urls
    text = re.sub(r'[^\w\s]', ' ', text) # remove punctuation
    text = re.sub(r'\d+', '', text)       # remove numbers
    text = ' '.join(text.split())
    return text

def main(args):
    print('Loading', args.input)
    df = pd.read_excel(args.input)
    print('Columns:', df.columns.tolist())

    # choose text column
    text_col = None
    for c in ['Eng_Trans_News_Body','Translated_News','News_Body','News_Title','title','text']:
        if c in df.columns:
            text_col = c
            break
    if text_col is None:
        # fallback: first column containing 'news' or 'title' or 'body'
        for c in df.columns:
            if any(k in c.lower() for k in ['news','title','body']):
                text_col = c
                break
    if text_col is None:
        raise SystemExit('No suitable text column found in dataset.')
    print('Using text column:', text_col)

    if 'Label' not in df.columns:
        raise SystemExit("Dataset must contain a 'Label' column.")

    df = df.dropna(subset=[text_col, 'Label'])
    df['text'] = df[text_col].astype(str).apply(clean_text)

    # encode labels robustly
    def encode_label(x):
        s = str(x).lower()
        if any(k in s for k in ['fake','false','misleading','fabricated']):
            return 1
        if any(k in s for k in ['true','real','genuine','authentic']):
            return 0
        # fallback: numeric?
        try:
            return int(x)
        except:
            return 1 if 'fake' in s else 0
    df['Label_enc'] = df['Label'].apply(encode_label)

    X = df['text']
    y = df['Label_enc']

    # vectorize with ngrams
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_features=20000)
    X_vec = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, stratify=y, random_state=42)

    # LinearSVC with balanced class weight
    model = LinearSVC(class_weight='balanced', max_iter=10000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print('Classification report:\n', classification_report(y_test, y_pred, target_names=['Real','Fake']))
    print('Confusion matrix:\n', confusion_matrix(y_test, y_pred))

    with open(args.out_model, 'wb') as f:
        pickle.dump(model, f)
    with open(args.out_vectorizer, 'wb') as f:
        pickle.dump(vectorizer, f)
    print('Saved model to', args.out_model)
    print('Saved vectorizer to', args.out_vectorizer)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--out-model', default='model.pkl')
    parser.add_argument('--out-vectorizer', default='vectorizer.pkl')
    args = parser.parse_args()
    main(args)
