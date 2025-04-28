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

import gym
from gym import spaces
import random
import pickle

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

class StockTradingEnv(gym.Env):
    """
    A simple stock trading environment for RL agent.
    The agent observes state features and decides to predict price movement.
    """
    metadata = {'render.modes': ['human']}
    
    def __init__(self, data):
        super(StockTradingEnv, self).__init__()
        self.data = data.reset_index(drop=True)
        self.current_step = 0
        self.total_steps = len(data) - 1
        
        # Actions: 0 = predict price down, 1 = predict price up
        self.action_space = spaces.Discrete(2)
        
        # Observation: open, high, low, volume (scaled)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(4,), dtype=np.float32
        )
        
        self.scaler = MinMaxScaler()
        self.scaled_data = self.scaler.fit_transform(self.data[['open', 'high', 'low', 'volume']])
        
    def reset(self):
        self.current_step = 0
        return self.scaled_data[self.current_step]
    
    def step(self, action):
        done = False
        reward = 0
        
        current_price = self.data.loc[self.current_step, 'close']
        next_price = self.data.loc[self.current_step + 1, 'close']
        
        price_diff = next_price - current_price
        
        # Reward: +1 if action matches price movement direction, else -1
        if (action == 1 and price_diff > 0) or (action == 0 and price_diff <= 0):
            reward = 1
        else:
            reward = -1
        
        self.current_step += 1
        if self.current_step >= self.total_steps:
            done = True
        
        next_state = self.scaled_data[self.current_step] if not done else np.zeros(self.observation_space.shape)
        
        info = {}
        
        return next_state, reward, done, info
    
    def render(self, mode='human'):
        pass

class RLPricePredictor:
    def __init__(self, symbol="TCB", episodes=1000, gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, alpha=0.01):
        self.symbol = symbol
        self.episodes = episodes
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.alpha = alpha
        
        self.env = None
        self.q_table = None
        self.trained = False
        
        self.model_file = f'rl_price_model_{self.symbol}.pkl'
        
    def fetch_training_data(self, symbol, days=365):
        end_date = datetime.today()
        start_date = end_date - timedelta(days=days)
        df = stock_historical_data(
            symbol=symbol,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )
        return df[['open', 'high', 'low', 'close', 'volume']].dropna()
    
    def build_env(self, data):
        self.env = StockTradingEnv(data)
        # Initialize Q-table with zeros: states x actions
        state_size = self.env.observation_space.shape[0]
        # Discretize state space for Q-table indexing
        self.bins = [10, 10, 10, 10]  # 10 bins per feature
        self.q_table = np.zeros(self.bins + [self.env.action_space.n])
        
    def discretize_state(self, state):
        ratios = []
        for i in range(len(state)):
            idx = int(state[i] * (self.bins[i] - 1))
            # Clip index to valid range
            idx = max(0, min(idx, self.bins[i] - 1))
            ratios.append(idx)
        return tuple(ratios)
    
    def train_model(self, symbol='TCB'):
        data = self.fetch_training_data(symbol)
        self.build_env(data)
        
        for e in range(self.episodes):
            state = self.env.reset()
            state_disc = self.discretize_state(state)
            done = False
            total_reward = 0
            
            while not done:
                if np.random.rand() <= self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(self.q_table[state_disc])
                
                next_state, reward, done, _ = self.env.step(action)
                next_state_disc = self.discretize_state(next_state)
                
                old_value = self.q_table[state_disc + (action,)]
                next_max = np.max(self.q_table[next_state_disc])
                
                # Q-learning update
                new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
                self.q_table[state_disc + (action,)] = new_value
                
                state_disc = next_state_disc
                total_reward += reward
            
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
            if (e+1) % 100 == 0:
                print(f"Episode {e+1}/{self.episodes}, Total Reward: {total_reward}, Epsilon: {self.epsilon:.4f}")
        
        # Save Q-table
        with open(self.model_file, 'wb') as f:
            pickle.dump(self.q_table, f)
        
        self.trained = True
    
    def load_model(self):
        if os.path.exists(self.model_file):
            with open(self.model_file, 'rb') as f:
                self.q_table = pickle.load(f)
            self.trained = True
            print(f"Loaded RL model for {self.symbol}")
        else:
            print(f"No RL model found for {self.symbol}")
    
    def predict_price(self, open, high, low, volume):
        if not self.trained:
            raise Exception("RL model not trained yet")
        
        state = np.array([open, high, low, volume])
        # Build env if not exists
        if self.env is None:
            data = self.fetch_training_data(self.symbol)
            self.build_env(data)
        
        # Scale state to [0,1]
        state_scaled = self.env.scaler.transform([state])[0]
        state_disc = self.discretize_state(state_scaled)
        
        action = np.argmax(self.q_table[state_disc])
        
        # Predict price movement based on action
        # action 1 = price up, 0 = price down
        # For demonstration, return current close price +/- a fixed delta
        current_price = (open + high + low) / 3
        delta = 0.01 * current_price  # 1% price change
        
        predicted_price = current_price + delta if action == 1 else current_price - delta
        return round(float(predicted_price), 2)
