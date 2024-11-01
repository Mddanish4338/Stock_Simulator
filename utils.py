



# # import yfinance as yf
# # import pandas as pd
# # from sklearn.model_selection import train_test_split
# # from sklearn.linear_model import LinearRegression

# # def get_stock_data(ticker_symbol):
# #     """Fetch daily stock data."""
# #     data = yf.download(ticker_symbol, period="1mo", interval="1d")
# #     if data.empty:
# #         raise ValueError("No data found for this ticker.")
# #     return data

# # def train_model(data):
# #     """Train a linear regression model on stock data."""
# #     data['Date'] = data.index
# #     data['Days'] = (data['Date'] - data['Date'].min()).dt.days
# #     X = data[['Days']]
# #     y = data['Close']

# #     # Train/test split
# #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# #     model = LinearRegression()
# #     model.fit(X_train, y_train)
    
# #     return model

# # def get_next_market_days(start_date, n_days):
# #     """Get the next n market days (Monday to Friday)."""
# #     market_days = []
# #     current_date = start_date

# #     while len(market_days) < n_days:
# #         if current_date.weekday() < 5:  # Monday to Friday
# #             market_days.append(current_date)
# #         current_date += pd.Timedelta(days=1)

# #     return market_days

# # def predict_next_days(model, last_day, n_days=3):
# #     """Predict stock prices for the next n market days."""
# #     predictions = []
    
# #     # Ensure last_day is a datetime object
# #     if isinstance(last_day, (int, float)):
# #         last_day = pd.Timestamp.now().normalize()  # Use current date if last_day is int or float
# #     elif isinstance(last_day, str):
# #         last_day = pd.Timestamp(last_day)
    
# #     market_days = get_next_market_days(last_day + pd.Timedelta(days=1), n_days)

# #     for day in market_days:
# #         days_since_start = (day - market_days[0]).days
# #         next_day = pd.DataFrame([[days_since_start]], columns=['Days'])  # Create DataFrame
# #         predicted_price = model.predict(next_day)
# #         predictions.append((day.date(), predicted_price[0]))  # Include date with prediction

# #     return predictions

# # def get_news(ticker_symbol):
# #     """Fetch news related to the stock using Yahoo Finance API."""
# #     stock = yf.Ticker(ticker_symbol)
# #     news_data = stock.news
# #     news_articles = []

# #     for article in news_data:
# #         news_articles.append({
# #             'title': article['title'],
# #             'url': article['link'],
# #             'description': article.get('summary', 'No description available')  # Use get to avoid KeyError
# #         })

# #     if not news_articles:
# #         raise ValueError("No news articles found for this stock.")
    
# #     return news_articles




# import yfinance as yf
# import pandas as pd
# from sklearn.linear_model import LinearRegression
# import numpy as np
# from datetime import timedelta, datetime

# def get_stock_data(ticker):
#     """Fetch historical stock data for a given ticker."""
#     data = yf.download(ticker, period='1y', interval='1d')
#     return data

# def train_model(data):
#     """Train a linear regression model using the stock data."""
#     data['Date'] = data.index
#     data['Date'] = pd.to_datetime(data['Date']).map(datetime.timestamp)
    
#     X = data['Date'].values.reshape(-1, 1)  # Features (dates)
#     y = data['Close'].values  # Target variable (closing prices)

#     model = LinearRegression()
#     model.fit(X, y)
#     return model

# def predict_next_days(model, last_day, n_days=3):
#     """Predict the next n days stock prices."""
#     market_days = get_next_market_days(last_day, n_days)
#     predictions = []

#     for day in market_days:
#         timestamp = pd.to_datetime(day).timestamp()
#         predicted_price = model.predict([[timestamp]])[0]
#         predictions.append((day.strftime('%Y-%m-%d'), predicted_price))

#     return predictions

# def get_next_market_days(start_date, n_days):
#     """Get the next n market days starting from a given date."""
#     current_date = start_date + timedelta(days=1)
#     market_days = []

#     while len(market_days) < n_days:
#         if current_date.weekday() < 5:  # Monday to Friday
#             market_days.append(current_date)
#         current_date += timedelta(days=1)

#     return market_days

# def get_news(stock_name):
#     """Fetch news articles related to a given stock."""
#     # Placeholder for news fetching logic
#     news_articles = []  # Your logic to fetch news articles
#     return news_articles

# def get_indices_data():
#     """Fetch latest data for major stock indices."""
#     indices = ['^NSEI', '^BSESN']  # Nifty 50 and Sensex
#     indices_data = {}
    
#     for index in indices:
#         data = yf.download(index, period="1d", interval="1m")
#         if not data.empty:
#             last_close = data['Close'].iloc[-1]
#             indices_data[index] = last_close
#         else:
#             indices_data[index] = "No data available"
    
#     return indices_data



import yfinance as yf
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta

def fetch_data(ticker, period='1mo'):
    data = yf.download(ticker, period=period)
    return data

def train_arima_model(data):
    # Ensure data index has a frequency
    data = data.asfreq('B')  # 'B' stands for business day frequency
    model = ARIMA(data['Close'], order=(5, 1, 0))
    model_fit = model.fit()
    return model_fit

def predict_next_days(model, data, last_day, n_days=3):
    # Create a DataFrame to hold the future dates
    future_dates = [last_day + timedelta(days=i) for i in range(1, n_days + 1)]
    future_dates_df = pd.DataFrame(index=future_dates)

    # Get the last known price
    last_price = model.forecast(steps=1).values[0]

    # Make predictions for the next n_days
    predictions = []
    for _ in range(n_days):
        # Use pd.concat instead of append
        data_with_last_price = pd.concat([data['Close'], pd.Series([last_price], index=[last_day + timedelta(days=_ + 1)])])
        model = ARIMA(data_with_last_price, order=(5, 1, 0))
        model_fit = model.fit()
        next_price = model_fit.forecast(steps=1).values[0]
        predictions.append(next_price)
        last_price = next_price  # Update last_price for the next iteration

    # Add predictions to the DataFrame
    future_dates_df['Predicted Price'] = predictions
    return future_dates_df

def get_next_market_days(start_date, n_days):
    market_days = []
    current_date = start_date
    while len(market_days) < n_days:
        if current_date.weekday() < 5:  # Monday to Friday are market days
            market_days.append(current_date)
        current_date += timedelta(days=1)
    return market_days
