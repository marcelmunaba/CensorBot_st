from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pandas as pd

def train_model(data):
    # Step 1: Feature extraction using CountVectorizer with unigrams and bigrams
    vectorizer = CountVectorizer(ngram_range=(1, 2))  # Include unigrams and bigrams
    X = vectorizer.fit_transform(data['text'])

    # Step 2: Assign weights to profane words based on severity ratings
    word_weights = dict(zip(data['text'], data['severity_rating']))
    word_weights = {word: weight for word, weight in word_weights.items() if weight > 1.0}  # Consider only profane words with severity > 1.0
    word_weights = {word: weight / max(word_weights.values()) for word, weight in word_weights.items()}  # Normalize weights

    # Step 3: Label the data based on severity
    y_label = (data['severity_rating'] >= 1.0).astype(int)

    # Step 4: Train Naive Bayes classifier with weighted features
    classifier = MultinomialNB()
    weighted_X = X.multiply([word_weights.get(word, 1.0) for word in vectorizer.get_feature_names_out()])
    classifier.fit(weighted_X, y_label)
    
    return classifier, vectorizer
