import streamlit as st
import pandas as pd

st.title('🏨 Hotel Machine Learning 🏨')

st.info('This is fisrt project machine learning with prediction')

# dropdown dataset
with st.expander('📁 Dataset Reservation'):
  df = pd.read_csv("https://raw.githubusercontent.com/arvisptka/Hotel_cancellationPrediction/refs/heads/main/Hotel%20Analysis%20Cancellation/datahotel_ec%2B.csv")
  df = df.drop(['meal','arrival_date_month','hotel', 'country', 'reserved_room_type', 'assigned_room_type'], axis = 1)
  df

  st.write(" Feature X ")
  X_Raw = df.drop('is_canceled', axis = 1)
  X_Raw

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

# Encode
encode = ['Grouping_country', 'reservation_status',  'market_segment',	'distribution_channel',	'deposit_type',	'customer_type']
df_hotels = pd.get_dummies(input_cshotel, prefix=encode)
df_hotels[:1]
