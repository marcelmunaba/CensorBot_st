import pandas as pd
from preprocessing import preprocess_text
from model import train_model
import numpy as np
import joblib
import re
import csv
import re
import random

def predict_curse(classifier, vectorizer, text, data):
    new_text = preprocess_text(text, data)
    #print("Preprocessed text: ", new_text) #debugging
    new_text_transformed = vectorizer.transform([new_text])
    prediction = classifier.predict(new_text_transformed)
    
    if prediction == 1:
        original_text = text
        # Get the feature names (words) from the vectorizer
        feature_names = vectorizer.get_feature_names_out()
        # Iterate through each word in the preprocessed text
        for word in new_text.split():
            # Check if the word is not in the feature names (learned during training)
            if word not in feature_names:
                continue
            # Replace the word with asterisks in the original text
            original_text = original_text.replace(word, "*" * len(word))
        
        print("Censored text: ", original_text)
        return 1, new_text
    else:
        return 0, new_text

def append_to_csv(file_path, new_word, severity_rating):
    with open(file_path, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # Round the severity rating to one decimal place
        severity_rating_rounded = round(severity_rating, 1)
        writer.writerow([new_word, severity_rating_rounded])  # Enclose both elements within a list or tuple

if __name__ == "__main__":
    #TODO: fetch from Database
    data = pd.read_csv('./profanity_sample.csv', encoding='utf-8')
    
    # Train the model
    classifier, vectorizer = train_model(data)
    
    print("Welcome to CensorBot :)")
    while True:
        # Test the model - Example : "What the heck is going on here?"
        text_input = input("Hello! What do have in your mind? : ")
        print("Testing the model with input: " + text_input)
        
        predict, preprocessedText = predict_curse(classifier, vectorizer, text_input, data)
        if (predict) == 1:
            print("Hey that's a bit rude! Watch your language >:|")
        else :
            print("I see. Congratulations on being polite :)")
        
        response = input("Do I get that right? (Y/n) ")
        new_resp = response.lower()
        if new_resp == 'n' and predict == 0:
            #write the new vocabulary to the csv
            append_to_csv('./profanity_sample.csv', preprocessedText, random.uniform(1.0, 2.0))
            print("Okay. I'll keep that in mind :)\n")
        elif new_resp == 'n' and predict == 1:
            #write the new vocabulary to the csv
            append_to_csv('./profanity_sample.csv', preprocessedText, random.uniform(0.0, 1.0))
            print("Okay. I'll keep that in mind :)\n")
        else:
            print("Thanks for your feedback!\n")