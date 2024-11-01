


# # import streamlit as st
# # import pandas as pd
# # import plotly.graph_objects as go
# # from datetime import datetime
# # from utils import fetch_data, train_arima_model, predict_next_days, get_next_market_days

# # def main():
# #     st.title("Stock Price Prediction App")

# #     ticker = st.text_input("Enter Stock Ticker Symbol (e.g., BHARTIARTL.NS):")

# #     if st.button("Predict"):
# #         # Fetch stock data
# #         data = fetch_data(ticker)
# #         if data.empty:
# #             st.error("No data found for the given ticker symbol. Please check the symbol and try again.")
# #             return

# #         # Display the last few rows of the data
# #         st.subheader("Historical Stock Prices")
# #         st.write(data.tail())

# #         # Train ARIMA model
# #         model = train_arima_model(data)

# #         # Get last day from the data
# #         last_day = data.index[-1]

# #         # Predict next market days
# #         n_days = 3  # Number of days to predict
# #         market_days = get_next_market_days(last_day + pd.Timedelta(days=1), n_days)

# #         # Predict prices for the next market days
# #         predictions = predict_next_days(model, data, last_day, n_days)

# #         # Prepare the predictions DataFrame
# #         predictions.index = market_days

# #         # Display predictions
# #         st.subheader("Predicted Prices for the Next Market Days")
# #         st.write(predictions)

# #         # Create candlestick chart for historical and predicted prices
# #         fig = go.Figure()

# #         # Candlestick for historical prices
# #         fig.add_trace(go.Candlestick(x=data.index,
# #                                       open=data['Open'],
# #                                       high=data['High'],
# #                                       low=data['Low'],
# #                                       close=data['Close'],
# #                                       name='Historical Prices'))

# #         # Add predicted prices to the chart
# #         fig.add_trace(go.Scatter(x=predictions.index, 
# #                                  y=predictions['Predicted Price'], 
# #                                  mode='lines+markers', 
# #                                  name='Predicted Prices', 
# #                                  line=dict(color='orange', width=2)))

# #         # Update layout
# #         fig.update_layout(title=f'Candlestick Chart for {ticker}',
# #                           xaxis_title='Date',
# #                           yaxis_title='Price',
# #                           xaxis_rangeslider_visible=False)

# #         # Display the chart in Streamlit
# #         st.plotly_chart(fig)

# # if __name__ == "__main__":
# #     main()



# import streamlit as st
# import pandas as pd
# import plotly.graph_objects as go
# from datetime import datetime
# from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.svm import SVR
# from xgboost import XGBRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, mean_absolute_error
# from utils import fetch_data, get_next_market_days

# def train_linear_regression(data):
#     features = data[['Open', 'High', 'Low', 'Volume']]
#     target = data['Close']
#     model = LinearRegression()
#     model.fit(features, target)
#     return model

# def train_random_forest(data):
#     features = data[['Open', 'High', 'Low', 'Volume']]
#     target = data['Close']
#     model = RandomForestRegressor(n_estimators=100, random_state=42)
#     model.fit(features, target)
#     return model

# def train_svr(data):
#     features = data[['Open', 'High', 'Low', 'Volume']]
#     target = data['Close']
#     model = SVR(kernel='rbf')
#     model.fit(features, target)
#     return model

# def train_xgboost(data):
#     features = data[['Open', 'High', 'Low', 'Volume']]
#     target = data['Close']
#     model = XGBRegressor()
#     model.fit(features, target)
#     return model

# def predict_next_days(model, data, n_days):
#     features = data[['Open', 'High', 'Low', 'Volume']].values[-1].reshape(1, -1)
#     predictions = []
#     for _ in range(n_days):
#         next_price = model.predict(features)
#         predictions.append(next_price[0])
#         # Simulate new features for next prediction
#         features[0][0] = next_price[0]  # Updating Open for next day's prediction
#         features[0][1] = next_price[0]  # Updating High
#         features[0][2] = next_price[0]  # Updating Low
#         features[0][3] = features[0][3]  # Keeping Volume constant
#     return predictions

# def main():
#     st.title("Stock Price Prediction App")

#     ticker = st.text_input("Enter Stock Ticker Symbol (e.g., BHARTIARTL.NS):")

#     if st.button("Predict"):
#         # Fetch stock data
#         data = fetch_data(ticker)
#         if data.empty:
#             st.error("No data found for the given ticker symbol. Please check the symbol and try again.")
#             return

#         # Display the last few rows of the data
#         st.subheader("Historical Stock Prices")
#         st.write(data.tail())

#         n_days = 3  # Number of days to predict
#         models = {
#             "Linear Regression": train_linear_regression(data),
#             "Random Forest": train_random_forest(data),
#             "Support Vector Regression": train_svr(data),
#             "XGBoost": train_xgboost(data)
#         }

#         predictions_dict = {}
#         for model_name, model in models.items():
#             predictions = predict_next_days(model, data, n_days)
#             predictions_dict[model_name] = predictions

#             # Calculate accuracy metrics
#             y_true = data['Close'].values[-n_days:]  # Get true prices for the last n days
#             mse = mean_squared_error(y_true, predictions)
#             mae = mean_absolute_error(y_true, predictions)

#             st.subheader(f"Predicted Prices using {model_name}:")
#             st.write(predictions)
#             st.write(f"Mean Squared Error: {mse:.2f}")
#             st.write(f"Mean Absolute Error: {mae:.2f}")

#         # Create candlestick chart for historical prices
#         fig = go.Figure()
#         fig.add_trace(go.Candlestick(x=data.index,
#                                       open=data['Open'],
#                                       high=data['High'],
#                                       low=data['Low'],
#                                       close=data['Close'],
#                                       name='Historical Prices'))

#         # Add predicted prices to the chart
#         for model_name, predictions in predictions_dict.items():
#             future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=n_days)
#             fig.add_trace(go.Scatter(x=future_dates, 
#                                      y=predictions, 
#                                      mode='lines+markers', 
#                                      name=f'Predicted Prices ({model_name})', 
#                                      line=dict(width=2)))

#         # Update layout
#         fig.update_layout(title=f'Candlestick Chart for {ticker}',
#                           xaxis_title='Date',
#                           yaxis_title='Price',
#                           xaxis_rangeslider_visible=False)

#         # Display the chart in Streamlit
#         st.plotly_chart(fig)

# if __name__ == "__main__":
#     main()



import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from utils import fetch_data, get_next_market_days

def train_arima(data):
    model = ARIMA(data['Close'], order=(5, 1, 0))  # You can adjust the order as needed
    model_fit = model.fit()
    return model_fit

def predict_arima(model, n_days):
    forecast = model.forecast(steps=n_days)
    return forecast

def train_linear_regression(data):
    features = data[['Open', 'High', 'Low', 'Volume']]
    target = data['Close']
    model = LinearRegression()
    model.fit(features, target)
    return model

def train_random_forest(data):
    features = data[['Open', 'High', 'Low', 'Volume']]
    target = data['Close']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(features, target)
    return model

def train_svr(data):
    features = data[['Open', 'High', 'Low', 'Volume']]
    target = data['Close']
    model = SVR(kernel='rbf')
    model.fit(features, target)
    return model

def train_xgboost(data):
    features = data[['Open', 'High', 'Low', 'Volume']]
    target = data['Close']
    model = XGBRegressor()
    model.fit(features, target)
    return model

def predict_next_days(model, data, n_days):
    features = data[['Open', 'High', 'Low', 'Volume']].values[-1].reshape(1, -1)
    predictions = []
    for _ in range(n_days):
        next_price = model.predict(features)
        predictions.append(next_price[0])
        # Simulate new features for next prediction
        features[0][0] = next_price[0]  # Updating Open for next day's prediction
        features[0][1] = next_price[0]  # Updating High
        features[0][2] = next_price[0]  # Updating Low
        features[0][3] = features[0][3]  # Keeping Volume constant
    return predictions

def main():
    st.title("Stock Price Prediction App")

    ticker = st.text_input("Enter Stock Ticker Symbol (e.g., BHARTIARTL.NS):")

    if st.button("Predict"):
        # Fetch stock data
        data = fetch_data(ticker)
        if data.empty:
            st.error("No data found for the given ticker symbol. Please check the symbol and try again.")
            return

        # Display the last few rows of the data
        st.subheader("Historical Stock Prices")
        st.write(data.tail())

        n_days = 3  # Number of days to predict
        models = {
            "ARIMA": train_arima(data),
            "Linear Regression": train_linear_regression(data),
            "Random Forest": train_random_forest(data),
            "Support Vector Regression": train_svr(data),
            "XGBoost": train_xgboost(data)
        }

        predictions_dict = {}
        for model_name, model in models.items():
            if model_name == "ARIMA":
                predictions = predict_arima(model, n_days)
            else:
                predictions = predict_next_days(model, data, n_days)

            predictions_dict[model_name] = predictions

            # Calculate accuracy metrics for non-ARIMA models
            if model_name != "ARIMA":
                y_true = data['Close'].values[-n_days:]  # Get true prices for the last n days
                mse = mean_squared_error(y_true, predictions)
                mae = mean_absolute_error(y_true, predictions)

                st.subheader(f"Predicted Prices using {model_name}:")
                st.write(predictions)
                st.write(f"Mean Squared Error: {mse:.2f}")
                st.write(f"Mean Absolute Error: {mae:.2f}")
            else:
                # For ARIMA, we do not have a direct y_true comparison for predictions
                st.subheader(f"Predicted Prices using {model_name}:")
                st.write(predictions)

        # Create candlestick chart for historical prices
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=data.index,
                                      open=data['Open'],
                                      high=data['High'],
                                      low=data['Low'],
                                      close=data['Close'],
                                      name='Historical Prices'))

        # Add predicted prices to the chart
        for model_name, predictions in predictions_dict.items():
            future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=n_days)
            fig.add_trace(go.Scatter(x=future_dates, 
                                     y=predictions, 
                                     mode='lines+markers', 
                                     name=f'Predicted Prices ({model_name})', 
                                     line=dict(width=2)))

        # Update layout
        fig.update_layout(title=f'Candlestick Chart for {ticker}',
                          xaxis_title='Date',
                          yaxis_title='Price',
                          xaxis_rangeslider_visible=False)

        # Display the chart in Streamlit
        st.plotly_chart(fig)

if __name__ == "__main__":
    main()
