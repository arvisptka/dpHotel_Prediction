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
  st.write('Lead time ADR')
  st.scatter_chart(data = df, x = 'lead_time', y = 'adr', color = 'is_canceled')

  st.write('Total Pengunjung ADR')
  st.scatter_chart(data = df, x = 'total_bermalam', y = 'adr', color = 'is_canceled')

# Data Preparation  
with st.sidebar:
  benua = st.selectbox('Grouping_country : Benua', ('Eropa', 'USA', 'Others', 'Asia'))
  market = st.selectbox('reservation_status', ('Check-Out', 'Canceled', 'No-Show'))	
  total_bermalam = st.slider('total_bermalam', 1, 15, 55)
  st.write("Total menginap", total_bermalam, "Malam")
  
