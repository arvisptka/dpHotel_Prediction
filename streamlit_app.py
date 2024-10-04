import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.title('🏨 Hotel Machine Learning 🏨')

st.info('This is fisrt project machine learning with prediction')

# dropdown dataset
with st.expander('📁 Dataset Reservation'):
  df = pd.read_csv("https://raw.githubusercontent.com/arvisptka/Hotel_cancellationPrediction/refs/heads/main/Hotel%20Analysis%20Cancellation/datahotel_ec%2B.csv")
  df = df.drop(['hotel','arrival_date_month','meal','country','reserved_room_type','assigned_room_type'], axis = 1)
  df
  
  st.write(" Feature X ")
  X_Raw = df.drop(['is_canceled', 'reservation_status_date'], axis = 1)
  X_Raw

  st.write(' Target Y')
  y = df.is_canceled
  y

with st.expander('📊 Data Visualization'):
  st.write('Lead time ADR')
  st.scatter_chart(data = df, x = 'lead_time', y = 'adr', color = 'is_canceled')

  st.write('Total Pengunjung ADR')
  st.scatter_chart(data = df, x = 'total_bermalam', y = 'adr', color = 'is_canceled')

# Input Features  
with st.sidebar:
  st.header('Input Features')
  Grouping_country = st.selectbox('Benua', ('Eropa', 'USA', 'Others', 'Asia'))
  reservation_status = st.selectbox('Status Reservasi', ('Check-Out', 'Canceled', 'No-Show'))	
  total_bermalam = st.slider('Total Mennginap', 1, 15, 55)
  st.write("Total menginap", total_bermalam, "Malam")

  #Create dataframe for the input feature
  data = {'Grouping_country': Grouping_country,
          'reservation_status': reservation_status,
          'total_bermalam': total_bermalam}
  input_df = pd.DataFrame(data, index=[0])
  input_cshotel = pd.concat([input_df, X_Raw], axis = 0)

with st.expander('Input Feature'):
  st.write('Input Hotel')
  input_df
  st.write('Combine Hotel Data')
  input_cshotel

# Data Preparation
# Encode X
encode = ['Grouping_country', 'reservation_status', 'market_segment', 'distribution_channel', 'deposit_type', 'customer_type']
df_hotels = pd.get_dummies(input_cshotel, columns=encode)
X = df_hotels[1:]
input_row = df_hotels[:1]

with st.expander('Data Preparation'):
  st.write('Encode X (Input Hotel')
  input_row
  st.write('Y Target')
  y

# Model Training 
## Train ML RandomForest
rf = RandomForestClassifier()
rf.fit(X, y)

## Apply model to make prediction
prediksi = rf.predict(input_row)
predik_proba = rf.predict_proba(input_row)

df_prediksi_proba = pd.DataFrame(predik_proba)
df_prediksi_proba.columns = ['Not Cancel','Cancel']
df_prediksi_proba.rename(columns = {0 : 'Not Cancel',
                                     1 : 'Cancel'})
df_prediksi_proba
