from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import os

app = Flask(__name__)

# Function to load the stock model
def load_stock_model(ticker_symbol):
    model_filename = f'Saved Models/{ticker_symbol}.h5'
    if os.path.exists(model_filename):
        return load_model(model_filename)
    else:
        return load_model('models/ns10_close.keras')

# Function to predict stock prices for the next month
def predict_stock_prices(ticker_symbol, model):
    # Fetch historical stock prices from Yahoo Finance for the last month
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.DateOffset(months=1)
    historical_data = yf.download(ticker_symbol, start=start_date, end=end_date)

    # Extract the 'Close' prices from the historical data
    close_prices = historical_data['Close']

    # Normalize the data using Min-Max scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    close_prices_scaled = scaler.fit_transform(close_prices.values.reshape(-1, 1))

    # Define sequence length for LSTM model
    seq_length = 10

    # Create sequences of data for LSTM model
    def create_sequences(data, seq_length):
        X = []
        for i in range(len(data) - seq_length):
            X.append(data[i:(i + seq_length), 0])
        return np.array(X)

    # Create sequences of data for LSTM model
    X = create_sequences(close_prices_scaled, seq_length)

    # Reshape input data to be [samples, time steps, features]
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Make predictions using the LSTM model
    predicted_scaled = model.predict(X)

    # Invert the predictions to the original scale
    predicted = scaler.inverse_transform(predicted_scaled)

    # Generate dates for the next month
    next_month_dates = pd.date_range(end_date + pd.DateOffset(days=1), periods=len(predicted))

    return next_month_dates, predicted

#### for trend line
def plot_stock_data(stock, start, end):
    # Download the data from Yahoo Finance
    data = yf.download(stock, start, end)

    # Calculate the moving average of the 5-day and 30-day periods
    data['MA_5'] = data['Close'].rolling(5).mean()
    data['MA_30'] = data['Close'].rolling(30).mean()

    # Detect crossover points
    data['Signal'] = 0
    data.loc[data['MA_5'] > data['MA_30'], 'Signal'] = 1  # Buy signal
    data.loc[data['MA_5'] < data['MA_30'], 'Signal'] = -1  # Sell signal

    # Plot the data and highlight crossover points
    plt.figure(figsize=(10, 6))
    plt.plot(data['Close'], label='Close Price', linewidth=2)
    plt.plot(data['MA_5'], label='5-day Moving Average')
    plt.plot(data['MA_30'], label='30-day Moving Average')

    # Highlight Buy and Sell signals
    plt.scatter(data.index[data['Signal'] == 1], data['Close'][data['Signal'] == 1], color='green', marker='^', label='Buy Signal')
    plt.scatter(data.index[data['Signal'] == -1], data['Close'][data['Signal'] == -1], color='red', marker='v', label='Sell Signal')

    plt.title(f"{stock} Price and Moving Averages with Crossovers")
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()

    # Convert the plot to a base64 encoded string
    img = BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    return plot_url

@app.route('/trendline', methods=['POST'])
def trendline():
    if request.method == 'POST':
        stock = request.form['stock']
        start = request.form['start']
        end = request.form['end']
        plot_url = plot_stock_data(stock, start, end)
        return render_template('result.html', plot_url=plot_url)
    else:
        return render_template('index.html')


# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for predicting stock prices
@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        # Get the input from the form
        ticker_symbol = request.form['ticker_symbol']

        # Load the model for the given stock symbol
        model = load_stock_model(ticker_symbol)

        # Predict stock prices
        dates, predicted_prices = predict_stock_prices(ticker_symbol, model)

        # Convert the plot to a base64 encoded string
        img = BytesIO()
        plt.figure(figsize=(10, 6))
        plt.plot(dates, predicted_prices, label='Predicted Close Price', marker='o')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.title(f'Stock Price Prediction for {ticker_symbol} Next Month')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(img, format='png')
        plt.close()
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()

        return render_template('result.html', plot_url=plot_url)
    else:
        # If a GET request is received, simply render the form page again
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
