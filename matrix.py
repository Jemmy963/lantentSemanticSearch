import re
from sklearn.feature_extraction.text import TfidfVectorizer

def clean_text(text):
    #prepares text for preprocessing
    text = text.lower()

    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()

def create_tfidf_matrix(documents):
    #converts the uploaded documents into a TF-IDF matrix.
    vectoriser = TfidfVectorizer(stop_words = "english")

    tfidf_matrix = vectoriser.fit_transform(documents)

    return vectoriser,tfidf_matrix