# Integrative-Stock-Trend-Predictor

This project introduces a revolutionary implementation of web scraping integrated with Spark for stock prediction. The methodology involves two main phases: web scraping of stock data from the NSE website and sentiment analysis on news data from CNBCtv18. By leveraging web scraping and machine learning techniques, this project aims to empower informed decision-making in stock trading.

### A. ID Extraction
- Extracts stock IDs from the NSE (National Stock Exchange) website.
- NSE is a leading stock exchange in India, facilitating transparent securities trading.

### B. Graph Data Extraction
- Utilizes web scraping to extract graph data for the retrieved stock IDs.
- Graph data containing timestamp and price values are stored in a CSV file.
- The XGBoost algorithm is employed for model training and prediction.

## Model Training and Prediction using XGBoost Algorithm

XGBoost (Extreme Gradient Boosting) is an ensemble learning algorithm known for its speed, efficiency, and versatility. It sequentially builds decision trees to correct errors made by previous models, making it suitable for classification and regression tasks in machine learning.

- The data is split into training and testing sets using a Timeseries Split variant of k-fold cross-validation.
- Cross-validation ensures robustness and reduces the risk of overfitting.
- Lag features are created by shifting time-series variables, enhancing prediction accuracy.
- Three lag intervals (5 minutes, 30 minutes, and 60 minutes) are introduced into the data.
- XGBoost algorithm efficiently predicts stock trends, producing accurate results.

## Sentiment Analysis and Sentiment Score

The second phase of the proposed model involves sentiment analysis on news data extracted from the CNBCtv18 website.

### D. News Scraping
- Web scraping extracts news data from the CNBCtv18 website.
- BeautifulSoup library is utilized for data extraction.

### E. Sentiment Analysis
- Apache Spark is employed for sentiment analysis.
- NLTK (Natural Language Toolkit) is used for text processing and sentiment analysis.
- Sentiment scores (-1, 0, 1) are generated based on the analysis of news articles.

## Buy/Sell Decision Based on Analysis

Based on the results obtained from stock prediction and sentiment analysis:
- If stock prediction and sentiment score are positive, the user is advised to buy the stock.
- If either prediction or sentiment score is negative, the user is advised to sell the stock.
- Transactions are recommended at the end of the day.

## Conclusion

The proposed methodology integrates web scraping, machine learning, and sentiment analysis to provide informed stock trading decisions. By unifying data retrieval and analysis techniques, this project offers a holistic approach to stock prediction and trading recommendations, showcasing the versatility of integrated technologies.

For detailed implementation and analysis, please refer to the provided documentation.
