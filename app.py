import nltk
from pyspark.sql.functions import when, col
from bs4 import BeautifulSoup
import bs4
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import os
import seaborn as sns
import xgboost as xgb
from xgboost import XGBRegressor
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import train_test_split
import streamlit as st
import requests
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
from nltk.sentiment import SentimentIntensityAnalyzer
import findspark
findspark.init()


nltk.download('vader_lexicon')
color_pal = sns.color_palette()
plt.style.use('fivethirtyeight')

st.set_option('deprecation.showPyplotGlobalUse', False)

ticker_symbol = ""
st.title("Stock Trend Prediction")


session = requests.session()

head = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}


def get_id(name):
    search_url = 'https://www.nseindia.com/api/search/autocomplete?q={}'
    get_details = 'https://www.nseindia.com/api/quote-equity?symbol={}'

    session.get('https://www.nseindia.com/', headers=head)

    search_results = session.get(url=search_url.format(name), headers=head)
    search_data = search_results.json()

    if 'symbols' in search_data and search_data['symbols']:
        search_result = search_data['symbols'][0]['symbol']

        company_details = session.get(
            url=get_details.format(search_result), headers=head)

        try:
            identifier = company_details.json()['info']['identifier']
            return identifier
        except KeyError:
            return f"Identifier not found for '{name}'"
    else:
        return f"No results found for '{name}'"


# Function to read historical data for multiple stocks
@st.cache_data
def read_stock_data(directory):
    stock_data = {}
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            stock_name = os.path.splitext(filename)[0]
            df = pd.read_csv(os.path.join(directory, filename))
            if 'datetime' in df.columns and 'close' in df.columns:
                stock_data[stock_name] = df[['datetime', 'close']]
    return stock_data

def train_model(stock_data):
    models = {}
    for stock_name, df in stock_data.items():
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['year'] = df['datetime'].dt.year
        df['month'] = df['datetime'].dt.month
        df['day'] = df['datetime'].dt.day
            
        X = df[['year', 'month', 'day']].values
        y = df['close']
            
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
        model = XGBRegressor()
        model.fit(X_train, y_train)
        models[stock_name] = model
            
    return models

def predict_price(model, date):
# Convert the date to a pandas datetime object
    date = pd.to_datetime(date)
        
    # Extract year, month, and day from the date
    year = date.year
    month = date.month
    day = date.day
        
    # Make prediction using the model
    prediction = model.predict(np.array([[year, month, day]]))[0]
        
    return prediction


# Read stock data from archive folder
archive_folder = "archive"
stock_data = read_stock_data(archive_folder)

# Select a stock for visualization
selected_stock = st.selectbox("Select a stock", list(stock_data.keys()))

# Display historical stock price data
if st.button("Get Stock Prediction"):
    st.subheader("Historical Stock Price Data")
    print(selected_stock)
    st.write(stock_data[selected_stock])

    # Plot historical stock prices
    if selected_stock in stock_data:
        df = stock_data[selected_stock]
        df['Date'] = pd.to_datetime(df['datetime'])
        plt.figure(figsize=(10, 6))
        plt.plot(df['Date'], df['close'])
        plt.xlabel("Date")
        plt.ylabel("Closing Price")
        plt.title(f"{selected_stock} Historical Stock Prices")
        st.pyplot()

    models = train_model(stock_data)
        
    today_date = datetime.date.today()

        # Predict stock price for today's EOD
    prediction = predict_price(models[selected_stock], today_date)

        # Display the predicted price
    st.subheader("Predicted Stock Price for Today's EOD")
    st.write(f"The predicted closing price for {selected_stock} on {today_date} is {prediction:.2f} based on Historical Data")

    company_name = selected_stock

    # Button to trigger the API call
    if company_name:
        ticker_symbol = get_id(company_name)
        st.write(
            f"The stock identifier for '{company_name}' is: {ticker_symbol}")

    stock_url = f'https://www.nseindia.com/api/chart-databyindex?index={ticker_symbol}'

    # Create a session
    session = requests.Session()

    # Define headers including the User-Agent
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    # Perform an initial request to the website to obtain cookies
    initial_response = session.get('https://www.nseindia.com/', headers=headers)
    print("Initial Response Status Code:", initial_response.status_code)

    # Perform the stock data request using the obtained session
    response = session.get(stock_url, headers=headers)
    print("Stock Data Response Status Code:", response.status_code)

    # Check if the request was successful before proceeding
    if response.ok:
        dayta = pd.DataFrame(response.json()['grapthData'])

        dayta.columns = ['timestamp', 'price']
        dayta['timestamp'] = pd.to_datetime(dayta['timestamp'], unit='ms')

        # Plotting
        dayta.plot(x='timestamp', y='price')
        plt.show()

        # Save data with timestamps in human-readable format to 'dayta.csv'
        dayta.to_csv('dayta.csv', index=False)

    else:
        print("Error fetching stock data.")


    # Intraday Stock Market
    st.subheader("Intraday Stock Market")
    df = pd.read_csv("dayta.csv")
    df = df.set_index('timestamp')
    df.index = pd.to_datetime(df.index)
    st.write(df.head(15))

    # Visualize the data
    st.subheader("Visualization of Data")
    fig = plt.figure(figsize=(12, 6))
    plt.plot(df.price)
    st.pyplot(fig)

    # Spliting into train and test
    # Train Test Split
    st.subheader("Train and Test Split")
    split_percentage = 0.75

    # Calculate the index for splitting
    split_index = int(len(df) * split_percentage)

    # Split the DataFrame
    train = df.iloc[:split_index]
    test = df.iloc[split_index:]

    # Plotting
    fig, ax = plt.subplots(figsize=(15, 5))
    train.plot(ax=ax, label='Training Set', color='blue',
            title='Data Train/Test Split')
    test.plot(ax=ax, label='Test Set', color='orange')
    # Use 'index' to get the timestamp from the index
    split_timestamp = df.index[split_index]
    ax.axvline(split_timestamp, color='black', ls='--', label='Train/Test Split')

    # Display the plot in Streamlit
    st.pyplot(fig)


    def create_features(df):
        """
        Create time series features based on time series index.
        """
        df = df.copy()
        df['second'] = df.index.second
        df['minute'] = df.index.minute
        df['hour'] = df.index.hour
        df['day'] = df.index.day

        return df


    df = create_features(df)

    # Adding the lag


    def add_lags(df):
        target_map = df['price'].to_dict()
        df['lag1'] = (df.index - pd.Timedelta('5 minutes')).map(target_map)
        df['lag2'] = (df.index - pd.Timedelta('30 minutes')).map(target_map)
        df['lag3'] = (df.index - pd.Timedelta('60 minutes')).map(target_map)
        return df


    df = add_lags(df)

    # Displaying Lag df
    st.subheader("First 5 rows of lag added Dataframe")
    st.write(df.head(5))

    st.subheader("Last 5 rows of lag added Dataframe")
    st.write(df.tail(5))

    st.subheader("Cross-Validation")

    tss = TimeSeriesSplit(n_splits=5, test_size=60, gap=5)
    fig, axs = plt.subplots(5, 1, figsize=(15, 15), sharex=True)

    fold = 0
    accuracies = []
    preds = []
    scores = []
    for train_idx, val_idx in tss.split(df):
        train = df.iloc[train_idx]
        test = df.iloc[val_idx]

        train = create_features(train)
        test = create_features(test)

        FEATURES = ['second', 'minute', 'hour', 'day', 'lag1', 'lag2', 'lag3']
        TARGET = 'price'

        X_train = train[FEATURES]
        y_train = train[TARGET]

        X_test = test[FEATURES]
        y_test = test[TARGET]

        reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree',
                            n_estimators=1000,
                            early_stopping_rounds=50,
                            objective='reg:linear',
                            max_depth=3,
                            learning_rate=1)
        reg.fit(X_train, y_train,
                eval_set=[(X_train, y_train), (X_test, y_test)],
                verbose=100)

        y_pred = reg.predict(X_test)
        preds.append(y_pred)
        score = np.sqrt(mean_squared_error(y_test, y_pred))
        scores.append(score)

        # Plotting
        train['price'].plot(ax=axs[fold], label='Training Set',
                            title=f'Data Train/Test Split Fold {fold}')
        test['price'].plot(ax=axs[fold], label='Test Set')
        ax.axvline(train.index.max(), color='black',
                ls='--', label='Train-Test Split')

        fold += 1

    # Display the plot
    st.pyplot(fig)

    # Plot mean scores
    st.write("RMSE Plot")
    fig_scores, ax_scores = plt.subplots(figsize=(10, 5))
    ax_scores.plot(scores, marker='o', linestyle='-', color='green')
    ax_scores.axhline(np.mean(scores), color='red',
                    linestyle='--', label='Mean Score')
    ax_scores.set_title('RMSEs across Folds')
    ax_scores.set_xlabel('Fold')
    ax_scores.set_ylabel('RMSE')
    ax_scores.legend()
    st.pyplot(fig_scores)

    # Print Mean RMSE and accuracy for each fold
    st.write(f'Mean RMSE across folds: {np.mean(scores):0.4f}')
    st.write(f'Fold RMSEs: {scores}')

    # Retrain on all data
    st.subheader("Retaining all data")
    df = create_features(df)

    FEATURES = ['second', 'minute', 'hour', 'day', 'lag1', 'lag2', 'lag3']
    TARGET = 'price'

    X_all = df[FEATURES]
    y_all = df[TARGET]

    reg = xgb.XGBRegressor(base_score=0.5,
                        booster='gbtree',
                        n_estimators=750,
                        objective='reg:linear',
                        max_depth=3,
                        learning_rate=1)
    reg.fit(X_all, y_all,
            eval_set=[(X_all, y_all)],
            verbose=100)
    reg.save_model("xgb.json")

    st.write(df.head(5))

    # Predicting the Future Trend
    start_timestamp = df.index[-1]
    stop_time = '15:30:00'

    start_timestamp = pd.to_datetime(start_timestamp)

    # Derive stop_timestamp by concatenating the date of start_timestamp with stop_time
    stop_timestamp = str(start_timestamp.date()) + ' ' + stop_time

    st.write("The Start time from which prediction starts: ", start_timestamp)
    st.write("The Start time from which prediction starts: ",
            pd.to_datetime(stop_timestamp))

    st.subheader("Creating the Timestamps")
    future = pd.date_range(start=start_timestamp,
                        end=stop_timestamp, freq='1min')
    future_df = pd.DataFrame(index=future)
    future_df = future_df.set_index(future)
    future_df['isFuture'] = True
    df['isFuture'] = True
    df_and_future = pd.concat([df, future_df])
    df_and_future = create_features(df_and_future)
    df_and_future = add_lags(df_and_future)

    st.write(df_and_future.tail())

    st.subheader("Prediction")
    future_w_features = df_and_future.query('isFuture').copy()
    future_w_features['pred'] = reg.predict(future_w_features[FEATURES])
    st.write(future_w_features.head())


    # Comparing end of day price with current price
    # Assuming 'value_to_compare' is the value you want to match in the 'column_name'
    value_to_compare = df.index[-1]  # Corrected to be a string


    # Find the row where the specified value is present in the specified column
    matching_row = df[df.index == value_to_compare]
    if not matching_row.empty:
        # Extract the first matching row (if there are multiple matches)
        matching_row = matching_row.iloc[0]

        # Access the 'pred' and 'price' columns for the selected row
        predicted_price = future_w_features['pred'].iloc[-1]
        current_price = matching_row['price']

        # Compare the predicted price with the actual price
        if predicted_price > current_price:
            action = 'Buy'
        else:
            action = 'Sell'

        # Display the results in Streamlit
        st.write(
            f"The end of day predicted price: {predicted_price}, current price: {current_price}")
        #st.write(f"Hence the current action: {action}")
    else:
        st.write(f"No matching row found for Timestamp={value_to_compare}")

    test['prediction'] = reg.predict(X_test)
    df = df.merge(test[['prediction']], how='left',
                left_index=True, right_index=True)


    # Plotting The Predictions

    def plot_predictions():
        fig, ax = plt.subplots(figsize=(10, 5))
        df['price'].plot(ax=ax, label='Original Data', color='blue')

        # Plotting the second DataFrame on the same axes
        test['prediction'].plot(
            ax=ax, label='Validation Predictions', color='orange', style='.', ms=1, lw=1)
        future_w_features['pred'].plot(
            ax=ax, label='Future Predictions', color='red', ms=1, lw=1)

        # Adding labels, title, and legend
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title(
            'Comparison of Original Data, Validation Predictions, and Future Predictions')
        plt.legend()

        return fig


    # Display the plot in Streamlit
    st.pyplot(plot_predictions())


    # Sentiment Analysis
    company_name = company_name.replace(' ', '-')
    url = f'https://www.cnbctv18.com/tags/{company_name}.htm'

    # Make a request to the website
    session = requests.Session()
    response = session.get(url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the HTML content
        soup = bs4.BeautifulSoup(response.text, 'html.parser')

        # Create empty lists to store titles and URLs
        titles = []
        urls = []

        # Find all anchor tags with the specified class
        anchors = soup.find_all('a', class_='jsx-7411095eac133b06')

        # Iterate through each anchor tag
        for anchor in anchors:
            # Check if the anchor tag has an h2 tag
            h2_tag = anchor.h2
            if h2_tag:
                # Extract title and URL
                title = h2_tag.get_text(strip=True)
                url = anchor['href']

                # Append to the lists
                titles.append(title)
                urls.append(url)

                # Print the extracted information
                print('Title:', title)
                print('URL:', url)
                print('--------------------')

        # Create a DataFrame
        data = {'Title': titles, 'URL': urls}
        df1 = pd.DataFrame(data)

        # Save DataFrame to CSV
        df1.to_csv('output.csv', index=False)
        st.write(df1.head())

    else:
        print(f"Failed to retrieve content. Status code: {response.status_code}")


    # Function to extract content from divs with class="jsx-1801027680 nart-para"

    def extract_div_content(url):
        try:
            full_url = url
            session = requests.Session()
            response = session.get(url)
            soup = BeautifulSoup(response.content, 'html.parser')

            # Find all divs containing 'nart-para' in the class attribute
            divs = soup.find_all('div', class_=lambda x: x and 'narticle-data' in x)

            return [div.get_text(strip=True) for div in divs]
        except Exception as e:
            print(f"Error fetching content from {url}: {e}")
            return None


    # Read the CSV file into a DataFrame
    df2 = pd.read_csv("output.csv")

    # Add a new column 'div_content' to store the list of extracted content
    df2['div_content'] = df2['URL'].apply(extract_div_content)

    # Save the DataFrame back to the CSV file
    df2.to_csv("output_with_paragraphs.csv", index=False)
    st.write(df2.head())
    print("Extraction completed and results saved to 'output_with_paragraphs.csv'.")

    # spark.conf.set("spark.python.worker.timeout", "600s")
    spark = SparkSession.builder.master("local[*]").appName("MyApp").getOrCreate()
    # spark.conf.set("spark.worker.timeout", "600")

    df3 = spark.read.csv("output_with_paragraphs.csv",
                        header=True, inferSchema=True)



    def analyze_sentiment(text):
        if text is None:
            return 'neutral'  # or any default sentiment you prefer
        sia = SentimentIntensityAnalyzer()
        sentiment_score = sia.polarity_scores(text)
        # Assign sentiment category based on compound score
        if sentiment_score['compound'] >= 0.05:
            return 'positive'
        elif sentiment_score['compound'] <= -0.05:
            return 'negative'
        else:
            return 'neutral'


    # Create a user-defined function (UDF) for sentiment analysis
    sentiment_udf = udf(analyze_sentiment, StringType())
    st.write(df3.columns)
    # Apply the UDF to the 'paragraphs' column and create a new 'sentiment' column
    df_with_sentiment = df3.withColumn('sentiment', sentiment_udf('div_content'))

    # Display the results
    # df_with_sentiment.select('div_content', 'sentiment').show(truncate=False)


    # Assign scores based on sentiment
    df_with_score = df_with_sentiment.withColumn(
        'score',
        when(col('sentiment') == 'negative', -1)
        .when(col('sentiment') == 'positive', 1)
        .otherwise(0)
    )

    # Display the results with scores
    st.write(df_with_score.select('div_content', 'sentiment', 'score'))

    # Calculate the mean score
    mean_score = df_with_score.agg({'score': 'mean'}).collect()[0][0]
    st.write(f"Mean Score: {mean_score}")
    if mean_score > 0.5:
        st.write("Buy")
    else:
        st.write("Sell")