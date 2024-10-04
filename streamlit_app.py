import streamlit as st
import pandas as pd

st.title('🏨 Hotel Machine Learning 🏨')

st.info('This is fisrt project machine learning with prediction')

# dropdown dataset
with st.expander('📁 Dataset Reservation'):
  df = pd.read_csv("https://raw.githubusercontent.com/arvisptka/Hotel_cancellationPrediction/refs/heads/main/Hotel%20Analysis%20Cancellation/datahotel_ec%2B.csv")
  df

  st.write(" Feature X ")
  X = df.drop('is_canceled', axis = 1)
  X

  st.write(' Target Y')
  y = df.is_canceled
  y

with st.expander('📊 Data Visualization'):
  st.scatter_chart(x = 'lead_time', y = ', color = 'is_canceled')
