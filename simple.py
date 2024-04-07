import streamlit as st
import requests
from bs4 import BeautifulSoup
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# Function to fetch historical stock data
def fetch_stock_data(symbol):
    stock_url = f'https://www.nseindia.com/api/chart-databyindex?index={symbol}'
    response = requests.get(stock_url)
    if response.ok:
        data = response.json()['grapthData']
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    else:
        return None

# Function to perform sentiment analysis on CNBC articles related to the company
def analyze_sentiment(company_name):
    company_name = company_name.replace(' ', '-')
    url = f'https://www.cnbctv18.com/tags/{company_name}.htm'
    session = requests.Session()
    response = session.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        divs = soup.find_all('div', class_=lambda x: x and 'nart-para' in x)
        content = ' '.join([div.get_text(strip=True) for div in divs])
        sia = SentimentIntensityAnalyzer()
        sentiment_score = sia.polarity_scores(content)
        return sentiment_score['compound']
    else:
        return None

# Streamlit UI
st.title("Stock Price Prediction with Sentiment Analysis")
company_name = st.text_input("Enter the name of the company:", "Tata Motors")
symbol = st.text_input("Enter the stock symbol:", "TATAMOTORS")

if st.button("Predict Closing Price"):
    # Fetch historical stock data
    stock_data = fetch_stock_data(symbol)
    if stock_data is not None:
        # Create features for XGBoost model
        stock_data['day'] = stock_data.index.day
        stock_data['month'] = stock_data.index.month
        stock_data['year'] = stock_data.index.year
        stock_data['sentiment'] = analyze_sentiment(company_name)
        
        # Split data into features and target variable
        X = stock_data[['day', 'month', 'year', 'sentiment']]
        y = stock_data['price']
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train XGBoost model
        model = XGBRegressor()
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        st.write(f"Root Mean Squared Error (RMSE): {rmse}")
        
        # Predict the closing price for today
        last_row = stock_data.iloc[[-1]]
        today_features = last_row[['day', 'month', 'year', 'sentiment']]
        today_prediction = model.predict(today_features)[0]
        
        st.write(f"Predicted closing price for today: {today_prediction}")
    else:
        st.write("Error fetching historical stock data.")
