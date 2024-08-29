import streamlit as st
import pandas as pd
import predict as pred
import nltk

# Download necessary NLTK data - important for preprocessing!
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

@st.cache_data
def welcome():
      with st.chat_message("assistant"):
        st.markdown("Hi there, I'm CensorBot. What do you have in mind?")
  
# MODEL PREPARATION
@st.cache_resource
def load_model_and_vectorizer():
    data = pd.read_csv("./profanity_sample.csv", encoding='utf-8')
    classifier, vectorizer = pred.train_model(data)
    return classifier, vectorizer, data

classifier, vectorizer, data = load_model_and_vectorizer()

st.title("CensorBot Demo")
st.html("<p>Welcome! In this demo, CensorBot will tell you if you are polite or not. Try typing what is on your mind down below 👇</p>")
welcome()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
    
# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    first_start = 0
    with st.chat_message("user"):
        st.markdown(prompt)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Predict using the model
    predict, preprocessedText = pred.predict_curse(classifier, vectorizer, prompt, data)
    st.write(f"Testing the model with input: {prompt}")
    st.write(f"Preprocessed text: {preprocessedText}")
    originalText = pred.censor_text(predict, preprocessedText, vectorizer)
    st.write(f"Censored Text: {originalText}")

    # Generate response based on prediction
    if predict == 1:
        response = "Hey that's a bit rude! Watch your language :angry:"
    else:
        response = "I see. Thank you for being polite :smile:"

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    
    # Recurring message
    with st.chat_message("assistant"):
        st.markdown("Hi there, I'm CensorBot. What do you have in mind?")
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.messages.append({"role": "assistant", "content": "Hello! What do you have in mind?"})