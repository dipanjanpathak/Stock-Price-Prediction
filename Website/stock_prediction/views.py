from django.shortcuts import render
from django.http import JsonResponse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
import datetime as dt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import os

import matplotlib
matplotlib.use('Agg')  


# Load the model
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'stock_dl_model.keras')
model = load_model(model_path)


def index(request):
    if request.method == 'POST':
        stock = request.POST.get('stock')
        if not stock:
            stock = 'TATAMOTORS.NS'  # Default stock if none is entered
        
        start = dt.datetime(2000, 1, 1)
        end = dt.datetime(2024, 10, 1)
        
        df = yf.download(stock, start=start, end=end)
        
        # Descriptive Data
        data_desc = df.describe()
        
        # Exponential Moving Averages
        ema20 = df.Close.ewm(span=20, adjust=False).mean()
        ema50 = df.Close.ewm(span=50, adjust=False).mean()
        ema100 = df.Close.ewm(span=100, adjust=False).mean()
        ema200 = df.Close.ewm(span=200, adjust=False).mean()

        # Data splitting
        data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
        data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])
        
        # Scaling data
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_training_array = scaler.fit_transform(data_training)
        
        # Prepare data for prediction
        past_100_days = data_training.tail(100)
        final_df = pd.concat([past_100_days,data_testing], ignore_index=True)
        input_data = scaler.fit_transform(final_df)
        
        x_test, y_test = [], []
        for i in range(100, input_data.shape[0]):
            x_test.append(input_data[i - 100:i])
            y_test.append(input_data[i, 0])
        x_test, y_test = np.array(x_test), np.array(y_test)

        # Make predictions
        y_predicted = model.predict(x_test)
        
        # Inverse scaling for predictions
        scaler = scaler.scale_
        scale_factor = 1 / scaler[0]
        y_predicted = y_predicted * scale_factor
        y_test = y_test * scale_factor
        
        # Plotting
        plt.style.use("fivethirtyeight")
        
        # Plot 1: EMA 20 & 50
        plt.figure(figsize=(12, 4))
        plt.plot(df.Close, 'y', label='Closing Price',linewidth=2)
        plt.plot(ema20, 'g', label='EMA 20', linewidth=2)
        plt.plot(ema50, 'r', label='EMA 50', linewidth=2)
        plt.title("Closing Price vs Time (20 & 50 Days EMA)")
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.legend()
        ema_chart_path = "static/ema_20_50.png"
        plt.savefig(ema_chart_path)
        plt.close()

        # Plot 2: EMA 100 & 200
        plt.figure(figsize=(12, 4))
        plt.plot(df.Close, 'y', label='Closing Price', linewidth=2)
        plt.plot(ema100, 'g', label='EMA 100', linewidth=2)
        plt.plot(ema200, 'r', label='EMA 200', linewidth=2)
        plt.title("Closing Price vs Time (100 & 200 Days EMA)")
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.legend()
        ema_chart_path_100_200 = "static/ema_100_200.png"
        plt.savefig(ema_chart_path_100_200)
        plt.close()

        # Plot 3: Prediction vs Original Trend
        plt.figure(figsize=(12, 4))  # Create a new figure with the desired size
        plt.plot(y_test, 'g', label="Original Price", linewidth=2)
        plt.plot(y_predicted, 'r', label="Predicted Price", linewidth=2)
        plt.title("Prediction vs Original Trend")
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.legend()
        prediction_chart_path = "static/stock_prediction.png"
        plt.savefig(prediction_chart_path)  # Save the figure as a PNG file
        plt.close()  # Close the figure to free up memory
        
        # Save dataset as CSV
        csv_file_path = f"static/{stock}_dataset.csv"
        df.to_csv(csv_file_path)

        return render(request, 'index.html', {
            'plot_path_ema_20_50': ema_chart_path,
            'plot_path_ema_100_200': ema_chart_path_100_200,
            'plot_path_prediction': prediction_chart_path,
            'data_desc': data_desc.to_html(classes='table table-bordered'),
            'dataset_link': csv_file_path
        })

    return render(request, 'index.html')
