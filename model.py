import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
from vnstock import stock_historical_data
from datetime import datetime, timedelta
import joblib
import numpy as np
import os

class PricePredictor:
    def __init__(self, symbol='TCB'):
        self.model = make_pipeline(
            MinMaxScaler(),
            LinearRegression()
        )
        self.trained = False
        self.symbol = symbol
        self.scaler = MinMaxScaler()
        
        # Check for existing model file
        model_file = f'price_model_{self.symbol}.pkl'
        scaler_file = f'scaler_{self.symbol}.pkl'
        try:
            self.model = joblib.load(model_file)
            self.scaler = joblib.load(scaler_file)
            self.trained = True
            print(f"Loaded existing model for {self.symbol}")
        except FileNotFoundError:
            print(f"No existing model found for {self.symbol}")
        
    def fetch_training_data(self, symbol='TCB', days=365):
        end_date = datetime.today()
        start_date = end_date - timedelta(days=days)
        df = stock_historical_data(
            symbol=symbol,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )
        return df[['open', 'high', 'low', 'close', 'volume']].dropna()
    
    def train_model(self, symbol='TCB'):
        data = self.fetch_training_data(symbol)
        X = data[['open', 'high', 'low', 'volume']]
        y = data['close']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"Model trained with MAE: {mae:,.0f}, R2 Score: {r2:.2f}")
        self.trained = True
        
        # Save model and scaler
        joblib.dump(self.model, f'price_model_{symbol}.pkl')
        joblib.dump(self.scaler, f'scaler_{symbol}.pkl')
        
    def predict_price(self, open, high, low, volume):
        if not self.trained:
            raise Exception("Model not trained yet")
            
        # Scale input features
        input_data = np.array([[open, high, low, volume]])
        input_scaled = self.scaler.transform(input_data)
        
        prediction = self.model.predict(input_scaled)[0]
        return round(float(prediction), 2)
