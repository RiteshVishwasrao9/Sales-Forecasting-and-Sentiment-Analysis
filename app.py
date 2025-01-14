import streamlit as st
import pickle
import pandas as pd
from prophet import Prophet
from textblob import TextBlob

# Load the trained Prophet model
with open('prophet_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Sentiment analysis function using TextBlob
def sentiment_analysis(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    
    if polarity > 0:
        sentiment = "Positive"
    elif polarity < 0:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    
    return sentiment, polarity


st.title('Sales Forecasting with Prophet and Sentiment Analysis')

# Sales Forecasting Section
st.header('Sales Forecasting')
input_dates = st.text_area("Enter dates to predict (comma-separated, e.g. '2023-01-01, 2023-01-02, 2023-01-03')")

if input_dates:

    input_dates = input_dates.split(',')
    input_dates = [date.strip() for date in input_dates]
    
    
    corrected_dates = []
    for date in input_dates:
    
        if len(date) == 4 and date.isdigit():  
            corrected_dates.append(date + "-01-01") 
        else:
            corrected_dates.append(date)


    df_input = pd.DataFrame(corrected_dates, columns=['ds'])
    df_input['ds'] = pd.to_datetime(df_input['ds'], errors='coerce') 
    
    if df_input['ds'].isnull().any():
        st.error("Invalid date format detected. Please enter dates in 'YYYY-MM-DD' format.")
    else:
        future = model.make_future_dataframe(periods=0)
        forecast = model.predict(future)

        st.write("Forecasted Sales:")
        st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])

        # Plot forecast
        st.write("Forecast Plot:")
        fig = model.plot(forecast)
        st.pyplot(fig)

# Sentiment Analysis Section
st.header('Sentiment Analysis')
user_input = st.text_area("Enter a text to analyze sentiment")

if user_input:
    sentiment, polarity = sentiment_analysis(user_input)
    st.write(f"Sentiment: {sentiment}")
    st.write(f"Polarity Score: {polarity:.2f}")
    
    # Display sentiment as color coded
    if sentiment == "Positive":
        st.markdown("<h3 style='color: green;'>Positive Sentiment</h3>", unsafe_allow_html=True)
    elif sentiment == "Negative":
        st.markdown("<h3 style='color: red;'>Negative Sentiment</h3>", unsafe_allow_html=True)
    else:
        st.markdown("<h3 style='color: gray;'>Neutral Sentiment</h3>", unsafe_allow_html=True)
