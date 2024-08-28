import streamlit as st
import pandas as pd
import numpy as np

#TODO: MIGRATE MINE FUNCTION DARI PREDICT.PY TO THIS FILE
st.text("This is a test streamlit app")
# Insert a chat message container.
with st.chat_message("user"):
    st.write("Hello 👋")
    st.line_chart(np.random.randn(30, 3))

# Display a chat input widget at the bottom of the app.
st.chat_input("Say something")