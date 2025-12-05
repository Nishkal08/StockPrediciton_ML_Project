import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from matplotlib import pyplot as plt
import yfinance as yf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

try:
    stock_name = input("Enter Stock Name: ").upper()
    df = yf.download(stock_name, start="1990-01-01", end=datetime.today().strftime('%Y-%m-%d'), auto_adjust=True)
    df.reset_index(inplace=True)
    print(f"\nTotal data points: {len(df)}")
    print(df.head())

    close_prices = df["Close"].values.reshape(-1, 1)
    
    # Split data: 80% train, 20% test
    split_idx = int(len(close_prices) * 0.8)
    train_data = close_prices[:split_idx]
    test_data = close_prices[split_idx:]
    
    print(f"\nTrain size: {len(train_data)}, Test size: {len(test_data)}")
    
    scaler = MinMaxScaler()
    scaled_train = scaler.fit_transform(train_data)
    scaled_test = scaler.transform(test_data)
    
    sequence_len = min(60, len(scaled_train) - 1)
    if sequence_len < 2:
        raise ValueError("Not enough data points for training!")
    
    batch_size = min(32, len(scaled_train) // 2)
    train_generator = TimeseriesGenerator(scaled_train, scaled_train, length=sequence_len, batch_size=batch_size)
    
    model = Sequential([
        LSTM(50, activation='relu', return_sequences=True, input_shape=(sequence_len, 1)),
        Dropout(0.2),
        LSTM(70, activation="relu", return_sequences=True),
        Dropout(0.3),
        LSTM(80, activation="relu"),
        Dropout(0.4),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    
    print("\n--- Training Model ---")
    history = model.fit(train_generator, epochs=20, verbose=1)
    
    print("\n" + "="*50)
    print("MODEL ACCURACY EVALUATION")
    print("="*50)
    
    test_predictions = []
    for i in range(sequence_len, len(scaled_test)):
        sequence = scaled_test[i-sequence_len:i]
        prediction = model.predict(sequence.reshape(1, sequence_len, 1), verbose=0)
        test_predictions.append(prediction[0, 0])
    
    test_predictions = scaler.inverse_transform(np.array(test_predictions).reshape(-1, 1))
    actual_prices = test_data[sequence_len:]
    
    mse = mean_squared_error(actual_prices, test_predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual_prices, test_predictions)
    r2 = r2_score(actual_prices, test_predictions)
    
    mape = np.mean(np.abs((actual_prices - test_predictions) / actual_prices)) * 100
    
    actual_direction = np.diff(actual_prices.flatten()) > 0
    pred_direction = np.diff(test_predictions.flatten()) > 0
    direction_accuracy = np.mean(actual_direction == pred_direction) * 100
    
    print(f"\nRegression Metrics:")
    print(f"  • RMSE (Root Mean Squared Error): {rmse:.2f}")
    print(f"  • MAE (Mean Absolute Error): {mae:.2f}")
    print(f"  • MAPE (Mean Absolute % Error): {mape:.2f}%")
    print(f"  • R² Score: {r2:.4f}")
    
    print(f"\nDirectional Accuracy:")
    print(f"  • Correct trend prediction: {direction_accuracy:.2f}%")
    
    print(f"\n Price Context:")
    print(f"  • Average actual price: {np.mean(actual_prices):.2f}")
    print(f"  • Average predicted price: {np.mean(test_predictions):.2f}")
    print(f"  • Max prediction error: {np.max(np.abs(actual_prices - test_predictions)):.2f}")
    
    print(f"\nModel Quality Assessment:")
    if mape < 5:
        print("   Excellent (MAPE < 5%)")
    elif mape < 10:
        print("  ✓ Good (MAPE < 10%)")
    elif mape < 20:
        print("  ⚠ Fair (MAPE < 20%)")
    else:
        print("   Poor (MAPE > 20%)")
    
    if r2 > 0.9:
        print(f"  Strong correlation (R² = {r2:.3f})")
    elif r2 > 0.7:
        print(f"  ✓ Moderate correlation (R² = {r2:.3f})")
    else:
        print(f"  ⚠ Weak correlation (R² = {r2:.3f})")
    

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    
    axes[0, 0].plot(df["Date"][:split_idx], train_data, label="Training data", alpha=0.7)
    test_dates = df["Date"][split_idx + sequence_len:]
    axes[0, 0].plot(test_dates, actual_prices, label="Actual (test)", c="blue")
    axes[0, 0].plot(test_dates, test_predictions, label="Predicted (test)", c="red", linestyle="--")
    axes[0, 0].set_xlabel("Date")
    axes[0, 0].set_ylabel("Price")
    axes[0, 0].set_title(f"{stock_name} - Train/Test Split")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    

    axes[0, 1].plot(test_dates, actual_prices, label="Actual", marker='o', markersize=3)
    axes[0, 1].plot(test_dates, test_predictions, label="Predicted", marker='x', markersize=3)
    axes[0, 1].set_xlabel("Date")
    axes[0, 1].set_ylabel("Price")
    axes[0, 1].set_title("Test Set Predictions (Zoomed)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    

    errors = actual_prices.flatten() - test_predictions.flatten()
    axes[1, 0].plot(test_dates, errors, color='red', alpha=0.6)
    axes[1, 0].axhline(y=0, color='black', linestyle='--', linewidth=1)
    axes[1, 0].fill_between(test_dates, errors, 0, alpha=0.3, color='red')
    axes[1, 0].set_xlabel("Date")
    axes[1, 0].set_ylabel("Prediction Error ($)")
    axes[1, 0].set_title("Prediction Errors Over Time")
    axes[1, 0].grid(True, alpha=0.3)
    

    axes[1, 1].plot(history.history['loss'])
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Loss")
    axes[1, 1].set_title("Training Loss Curve")
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

    proceed = input("\n Warning: Future predictions compound errors. Continue? (y/n): ")
    if proceed.lower() == 'y':
        future_days = int(input("Days to predict: "))
        future_predictions = []
        last_sequence = scaled_data[-sequence_len:]
        
        for _ in range(future_days):
            prediction = model.predict(last_sequence.reshape(1, sequence_len, 1), verbose=0)
            future_predictions.append(prediction[0, 0])
            last_sequence = np.append(last_sequence[1:], prediction[0, 0])
        
        future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
        
        last_date = pd.to_datetime(df['Date'].iloc[-1])
        future_dates = pd.date_range(start=last_date, periods=future_days + 1)
        future_df = pd.DataFrame({"Date": future_dates[1:], "Price": future_predictions.flatten()})
        
        print("\n Future Predictions:")
        print(future_df)
        
        plt.figure(figsize=(12, 6))
        plt.plot(df["Date"][-100:], df["Close"][-100:], label="Historical", c="blue")
        plt.plot(future_df["Date"], future_df["Price"], label="Future Prediction", c="red", linestyle="--", marker='o')
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.title(f"{stock_name} - Future Price Prediction (Uncertainty Increases!)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

except Exception as e:
    print(f"\n Error occurred: {e}")
    print("\nStock name not found or insufficient data!")
    print("Please enter a valid stock ticker (e.g., AAPL, RELIANCE.NS)")