import re
import string
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import wordninja
import pandas

def preprocess_text(text, data):
    # Convert text to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    
    # Filter the DataFrame to include rows where severity rating >= 1.0
    filtered_data = data[data['severity_rating'] >= 1.0]

    # Extract the text column from the filtered DataFrame
    custom_dictionary = filtered_data['text']
    # Extract unique words from the text column and construct custom_dictionary
    custom_dictionary = set(custom_dictionary.str.split().explode())
    # Handle numerical digits followed by letters

    # Find all digits in the string
    digit_list = re.findall('\d', text)

    # Check if the digit 1 is in the list of digits
    if '0' in digit_list:
        text = re.sub(r'0+', 'o', text)
    if '1' in digit_list:
        text = re.sub(r'1+', 'i', text)
    if '3' in digit_list:
        text = re.sub(r'3+', 'e', text)
    if '4' in digit_list:
        text = re.sub(r'4+', 'a', text)
    if '5' in digit_list:
        text = re.sub(r'(?<!\d)5(?!\d)', 's', text)
    if '6' in digit_list or '9' in digit_list:
        text = re.sub(r'6+', 'g', text)

    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    for word in tokens :
        if word not in custom_dictionary:
            # Lemmatization
            lemmatizer = WordNetLemmatizer()
            tokens = [lemmatizer.lemmatize(word) for word in tokens]
            break

    # Split concatenated words
    for i, word in enumerate(tokens):
        if word not in custom_dictionary:
            split_words = wordninja.split(word)
            tokens[i:i+1] = split_words
    # Join the tokens back into a single string
    preprocessed_text = ' '.join(tokens)
    
    return preprocessed_text
