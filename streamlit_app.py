import streamlit as st
import pandas as pd

st.title('🏨 Hotel Machine Learning 🏨')

st.info('This is fisrt project machine learning with prediction')

# load dataset
df = pd.read_csv("https://raw.githubusercontent.com/arvisptka/Hotel_cancellationPrediction/refs/heads/main/Hotel%20Analysis%20Cancellation/datahotel_ec%2B.csv")
df
