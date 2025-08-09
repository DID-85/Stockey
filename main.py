

import os
import sys
import warnings
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import pymssql
import yfinance as yf
import talib
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
from prophet import Prophet
import optuna
from optuna.samplers import TPESampler
import logging

warnings.filterwarnings('ignore')

# =============================================
# CONFIGURATION
# =============================================


class TrainingConfig:
    # Database configuration for SQL Server
    DB_SERVER = 'localhost'
    DB_NAME = 'StockyDB'
    DB_USER = 'TEST'
    DB_PASSWORD = 'TEST'
    
    # Model paths
    MODEL_PATH = './models/'
    
    # Training parameters
    LOOKBACK_DAYS = 365
    FORECAST_HORIZON = 5  # Days
    TRAIN_TEST_SPLIT = 0.8
    VALIDATION_SPLIT = 0.2
    
    # Feature engineering
    TECHNICAL_INDICATORS = [
        'rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower',
        'ema_9', 'ema_21', 'adx', 'obv', 'atr'
    ]
    
    # Model hyperparameters (defaults)
    XGBOOST_PARAMS = {
        'n_estimators': 100,
        'max_depth': 7,
        'learning_rate': 0.01,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    }
    
    RF_PARAMS = {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2
    }

config = TrainingConfig()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =============================================
# DATABASE CONNECTION
# =============================================

def get_db_connection():
    """Create database connection using pymssql"""
    try:
        conn = pymssql.connect(
            server=config.DB_SERVER,
            user=config.DB_USER,
            password=config.DB_PASSWORD,
            database=config.DB_NAME
        )
        logger.info("Database connection established")
        return conn
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        sys.exit(1)

# =============================================
# DATA COLLECTION
# =============================================

class DataCollector:
    def __init__(self):
        self.conn = get_db_connection()
    
    def __del__(self):
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()
    
    def fetch_stock_list(self):
        """Get list of stocks to train on using cursor"""
        query = """
        SELECT DISTINCT symbol, name, token 
        FROM stocks 
        WHERE is_active = 1 
        AND symbol NOT IN ('NIFTY', 'BANKNIFTY')
        """
        try:
            with self.conn.cursor(as_dict=True) as cursor:
                cursor.execute(query)
                data = cursor.fetchall()
                return pd.DataFrame(data)
        except Exception as e:
            logger.error(f"Error fetching stock list: {e}")
            return pd.DataFrame()
    def fetch_historical_data(self, symbol, days=365):
    """Fetch historical data from database or Yahoo Finance"""
    try:
        # Try database first with cursor
        query = f"""
        SELECT timestamp, [open_price] as [open], [high_price] as [high], 
               [low_price] as [low], [close_price] as [close], [volume]
        FROM market_data md
        JOIN stocks s ON md.stock_id = s.stock_id
        WHERE s.symbol = '{symbol}'
        AND timestamp >= DATEADD(day, -{days}, GETDATE())
        ORDER BY timestamp ASC
        """
        
        with self.conn.cursor(as_dict=True) as cursor:
            cursor.execute(query)
            data = cursor.fetchall()
            df = pd.DataFrame(data)
        
        if df.empty or len(df) < 100:
            # Fallback to Yahoo Finance
            logger.info(f"Fetching {symbol} from Yahoo Finance as fallback")
            return self.fetch_from_yfinance(symbol, days)
        
        return df
        
    except Exception as e:
        logger.error(f"Error fetching data for {symbol} from database: {e}")
        logger.info(f"Attempting to fetch {symbol} from Yahoo Finance")
        return self.fetch_from_yfinance(symbol, days)

    def fetch_from_yfinance(self, symbol, days):
    """Fetch data from Yahoo Finance"""
    try:
        ticker = yf.Ticker(f"{symbol}.NS")
        df = ticker.history(period=f"{days}d")
        if df.empty:
            logger.warning(f"No data available from Yahoo Finance for {symbol}")
            return pd.DataFrame()
            
        df.reset_index(inplace=True)
        df.columns = [col.lower() for col in df.columns]
        df.rename(columns={'date': 'timestamp'}, inplace=True)
        return df
    except Exception as e:
        logger.error(f"Error fetching data from Yahoo Finance for {symbol}: {e}")
        return pd.DataFrame()   


    
    def save_to_database(self, df, symbol):
        """Save processed data to database using cursor"""
        try:
            with self.conn.cursor() as cursor:
                # Get stock_id
                stock_id_query = f"SELECT stock_id FROM stocks WHERE symbol = '{symbol}'"
                cursor.execute(stock_id_query)
                stock_id = cursor.fetchone()[0]
                
                # Prepare data for insertion
                for _, row in df.iterrows():
                    insert_query = """
                    INSERT INTO market_data_processed (
                        stock_id, timestamp, [open], [high], [low], [close], [volume]
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """
                    cursor.execute(insert_query, (
                        stock_id,
                        row['timestamp'],
                        row.get('open', 0),
                        row.get('high', 0),
                        row.get('low', 0),
                        row.get('close', 0),
                        row.get('volume', 0)
                    ))
                
                self.conn.commit()
                logger.info(f"Data saved for {symbol}")
                
        except Exception as e:
            logger.error(f"Error saving data for {symbol}: {e}")
            self.conn.rollback()

# =============================================
# FEATURE ENGINEERING
# =============================================

class FeatureEngineer:
    @staticmethod
    def calculate_technical_indicators(df):
        """Calculate all technical indicators"""
        if df.empty:
            return df
        
        try:
            # Price columns
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            volume = df['volume'].values.astype(float)
            
            # Trend Indicators
            df['sma_5'] = talib.SMA(close, timeperiod=5)
            df['sma_20'] = talib.SMA(close, timeperiod=20)
            df['sma_50'] = talib.SMA(close, timeperiod=50)
            df['ema_9'] = talib.EMA(close, timeperiod=9)
            df['ema_21'] = talib.EMA(close, timeperiod=21)
            
            # Momentum Indicators
            df['rsi'] = talib.RSI(close, timeperiod=14)
            df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(close)
            df['stoch_k'], df['stoch_d'] = talib.STOCH(high, low, close)
            
            # Volatility Indicators
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(close)
            df['atr'] = talib.ATR(high, low, close, timeperiod=14)
            
            # Volume Indicators
            df['obv'] = talib.OBV(close, volume)
            df['ad'] = talib.AD(high, low, close, volume)
            
            # Trend Strength
            df['adx'] = talib.ADX(high, low, close, timeperiod=14)
            df['plus_di'] = talib.PLUS_DI(high, low, close, timeperiod=14)
            df['minus_di'] = talib.MINUS_DI(high, low, close, timeperiod=14)
            
            # Price Features
            df['high_low_pct'] = (df['high'] - df['low']) / df['close'] * 100
            df['close_open_pct'] = (df['close'] - df['open']) / df['open'] * 100
            
            # Lag Features
            for i in [1, 2, 3, 5, 10]:
                df[f'close_lag_{i}'] = df['close'].shift(i)
                df[f'volume_lag_{i}'] = df['volume'].shift(i)
            
            # Rolling Statistics
            for window in [5, 10, 20]:
                df[f'close_rolling_mean_{window}'] = df['close'].rolling(window=window).mean()
                df[f'close_rolling_std_{window}'] = df['close'].rolling(window=window).std()
                df[f'volume_rolling_mean_{window}'] = df['volume'].rolling(window=window).mean()
            
            # Support and Resistance Levels
            df['resistance'] = df['high'].rolling(window=20).max()
            df['support'] = df['low'].rolling(window=20).min()
            df['pivot'] = (df['high'] + df['low'] + df['close']) / 3
            
            # Clean NaN values
            df.fillna(method='ffill', inplace=True)
            df.fillna(method='bfill', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return df
    
    @staticmethod
    def create_target_variables(df, horizon=1):
        """Create target variables for prediction"""
        df['target'] = df['close'].shift(-horizon)
        df['target_direction'] = (df['target'] > df['close']).astype(int)
        df['target_return'] = (df['target'] - df['close']) / df['close'] * 100
        return df

# =============================================
# MODEL TRAINING
# =============================================

class ModelTrainer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {}
        
    def prepare_data(self, df):
        """Prepare data for training"""
        # Select features
        feature_cols = [col for col in df.columns if col not in [
            'timestamp', 'target', 'target_direction', 'target_return',
            'open', 'high', 'low', 'close', 'volume'
        ]]
        
        # Remove rows with NaN in target
        df_clean = df.dropna(subset=['target'])
        
        X = df_clean[feature_cols]
        y = df_clean['target']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y, feature_cols
    
    def train_xgboost(self, X_train, y_train, X_val, y_val, use_optuna=True):
        """Train XGBoost model with optional hyperparameter tuning"""
        
        if use_optuna:
            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'gamma': trial.suggest_float('gamma', 0, 0.5),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0),
                }
                
                model = xgb.XGBRegressor(**params, random_state=42, n_jobs=-1)
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=False)
                
                predictions = model.predict(X_val)
                mse = mean_squared_error(y_val, predictions)
                return mse
            
            study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
            study.optimize(objective, n_trials=50, show_progress_bar=True)
            
            best_params = study.best_params
            logger.info(f"Best XGBoost params: {best_params}")
            
            model = xgb.XGBRegressor(**best_params, random_state=42, n_jobs=-1)
        else:
            model = xgb.XGBRegressor(**config.XGBOOST_PARAMS, random_state=42, n_jobs=-1)
        
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=False)
        
        return model
    
    def train_random_forest(self, X_train, y_train):
        """Train Random Forest model"""
        model = RandomForestRegressor(**config.RF_PARAMS, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        return model
    
    def train_prophet(self, df):
        """Train Prophet model for time series forecasting"""
        try:
            # Prepare data for Prophet
            prophet_df = df[['timestamp', 'close']].copy()
            prophet_df.columns = ['ds', 'y']
            prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
            
            # Initialize Prophet with custom parameters
            model = Prophet(
                changepoint_prior_scale=0.05,
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                seasonality_mode='multiplicative',
                interval_width=0.95
            )
            
            # Add custom seasonalities
            model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
            model.add_seasonality(name='quarterly', period=91.25, fourier_order=5)
            
            # Add Indian holidays
            model.add_country_holidays(country_name='IN')
            
            # Fit model
            model.fit(prophet_df)
            
            return model
            
        except Exception as e:
            logger.error(f"Error training Prophet: {e}")
            return None
    
    def create_ensemble_prediction(self, predictions_dict, weights=None):
        """Create weighted ensemble prediction"""
        if weights is None:
            weights = {
                'xgboost': 0.4,
                'random_forest': 0.2,
                'prophet': 0.4
            }
        
        ensemble_pred = np.zeros_like(predictions_dict['xgboost'])
        
        for model_name, pred in predictions_dict.items():
            if model_name in weights:
                ensemble_pred += pred * weights[model_name]
        
        return ensemble_pred

# =============================================
# BACKTESTING
# =============================================

class Backtester:
    @staticmethod
    def calculate_metrics(y_true, y_pred):
        """Calculate performance metrics"""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        
        # Directional accuracy
        direction_true = np.diff(y_true) > 0
        direction_pred = np.diff(y_pred) > 0
        directional_accuracy = np.mean(direction_true == direction_pred)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mape': mape,
            'directional_accuracy': directional_accuracy
        }
    
    @staticmethod
    def trading_simulation(df, predictions, initial_capital=100000):
        """Simulate trading based on predictions"""
        capital = initial_capital
        position = 0
        trades = []
        
        for i in range(len(predictions) - 1):
            current_price = df.iloc[i]['close']
            predicted_change = predictions[i + 1] - predictions[i]
            
            if predicted_change > current_price * 0.01:  # Buy signal (1% threshold)
                if position == 0:
                    position = capital / current_price
                    capital = 0
                    trades.append({'type': 'BUY', 'price': current_price, 'quantity': position})
                    
            elif predicted_change < -current_price * 0.01:  # Sell signal
                if position > 0:
                    capital = position * current_price
                    position = 0
                    trades.append({'type': 'SELL', 'price': current_price, 'capital': capital})
        
        # Close final position
        if position > 0:
            capital = position * df.iloc[-1]['close']
        
        total_return = (capital - initial_capital) / initial_capital * 100
        
        return {
            'total_return': total_return,
            'final_capital': capital,
            'num_trades': len(trades),
            'trades': trades
        }

# =============================================
# MAIN TRAINING PIPELINE
# =============================================

class TrainingPipeline:
    def __init__(self):
        self.data_collector = DataCollector()
        self.feature_engineer = FeatureEngineer()
        self.model_trainer = ModelTrainer()
        self.backtester = Backtester()
        
        # Create models directory
        os.makedirs(config.MODEL_PATH, exist_ok=True)
    
    def run(self, symbols=None):
        """Run complete training pipeline"""
        logger.info("Starting training pipeline...")
        
        # Get stock list
        if symbols is None:
            stocks_df = self.data_collector.fetch_stock_list()
            symbols = stocks_df['symbol'].tolist()[:10]  # Train on top 10 stocks
        
        all_models = {}
        all_metrics = {}
        
        for symbol in symbols:
            logger.info(f"Training models for {symbol}")
            
            try:
                # Fetch data
                df = self.data_collector.fetch_historical_data(symbol, config.LOOKBACK_DAYS)
                
                if df.empty or len(df) < 100:
                    logger.warning(f"Insufficient data for {symbol}")
                    continue
                
                # Feature engineering
                df = self.feature_engineer.calculate_technical_indicators(df)
                df = self.feature_engineer.create_target_variables(df, config.FORECAST_HORIZON)
                
                # Prepare data
                X, y, feature_cols = self.model_trainer.prepare_data(df)
                
                # Train-test split
                split_idx = int(len(X) * config.TRAIN_TEST_SPLIT)
                X_train, X_test = X[:split_idx], X[split_idx:]
                y_train, y_test = y[:split_idx], y[split_idx:]
                
                # Validation split
                val_idx = int(len(X_train) * (1 - config.VALIDATION_SPLIT))
                X_train_final, X_val = X_train[:val_idx], X_train[val_idx:]
                y_train_final, y_val = y_train[:val_idx], y_train[val_idx:]
                
                # Train models
                logger.info(f"Training XGBoost for {symbol}")
                xgb_model = self.model_trainer.train_xgboost(X_train_final, y_train_final, X_val, y_val)
                
                logger.info(f"Training Random Forest for {symbol}")
                rf_model = self.model_trainer.train_random_forest(X_train_final, y_train_final)
                
                logger.info(f"Training Prophet for {symbol}")
                prophet_model = self.model_trainer.train_prophet(df)
                
                # Generate predictions
                xgb_pred = xgb_model.predict(X_test)
                rf_pred = rf_model.predict(X_test)
                
                # Prophet predictions
                if prophet_model:
                    future = prophet_model.make_future_dataframe(periods=len(X_test))
                    prophet_forecast = prophet_model.predict(future)
                    prophet_pred = prophet_forecast['yhat'].values[-len(X_test):]
                else:
                    prophet_pred = xgb_pred  # Fallback
                
                # Ensemble predictions
                predictions = {
                    'xgboost': xgb_pred,
                    'random_forest': rf_pred,
                    'prophet': prophet_pred
                }
                
                ensemble_pred = self.model_trainer.create_ensemble_prediction(predictions)
                
                # Calculate metrics
                metrics = self.backtester.calculate_metrics(y_test.values, ensemble_pred)
                
                # Trading simulation
                test_df = df.iloc[split_idx:].reset_index(drop=True)
                trading_results = self.backtester.trading_simulation(test_df, ensemble_pred)
                
                metrics.update(trading_results)
                
                logger.info(f"Metrics for {symbol}:")
                logger.info(f"  RMSE: {metrics['rmse']:.2f}")
                logger.info(f"  MAPE: {metrics['mape']:.2%}")
                logger.info(f"  Directional Accuracy: {metrics['directional_accuracy']:.2%}")
                logger.info(f"  Trading Return: {metrics['total_return']:.2%}")
                
                # Save models
                model_data = {
                    'symbol': symbol,
                    'xgboost': xgb_model,
                    'random_forest': rf_model,
                    'prophet': prophet_model,
                    'scaler': self.model_trainer.scaler,
                    'feature_cols': feature_cols,
                    'metrics': metrics
                }
                
                joblib.dump(model_data, f"{config.MODEL_PATH}{symbol}_models.pkl")
                
                all_models[symbol] = model_data
                all_metrics[symbol] = metrics
                
            except Exception as e:
                logger.error(f"Error training {symbol}: {e}")
                continue
        
        # Save ensemble model
        ensemble_data = {
            'models': all_models,
            'metrics': all_metrics,
            'trained_date': datetime.now(),
            'config': config.__dict__
        }
        
        joblib.dump(ensemble_data, f"{config.MODEL_PATH}ensemble_model.pkl")
        
        logger.info("Training pipeline completed!")
        
        # Generate report
        self.generate_report(all_metrics)
        
        return ensemble_data
    
    def generate_report(self, metrics):
        """Generate training report"""
        report = "\n" + "=" * 60 + "\n"
        report += "STOCKY AI MODEL TRAINING REPORT\n"
        report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += "=" * 60 + "\n\n"
        
        for symbol, metric in metrics.items():
            report += f"\n{symbol}:\n"
            report += f"  RMSE: {metric['rmse']:.2f}\n"
            report += f"  MAPE: {metric['mape']:.2%}\n"
            report += f"  Directional Accuracy: {metric['directional_accuracy']:.2%}\n"
            report += f"  Trading Return: {metric['total_return']:.2%}\n"
            report += f"  Number of Trades: {metric['num_trades']}\n"
        
        # Calculate average metrics
        avg_rmse = np.mean([m['rmse'] for m in metrics.values()])
        avg_accuracy = np.mean([m['directional_accuracy'] for m in metrics.values()])
        avg_return = np.mean([m['total_return'] for m in metrics.values()])
        
        report += "\n" + "=" * 60 + "\n"
        report += "AVERAGE PERFORMANCE:\n"
        report += f"  Average RMSE: {avg_rmse:.2f}\n"
        report += f"  Average Directional Accuracy: {avg_accuracy:.2%}\n"
        report += f"  Average Trading Return: {avg_return:.2%}\n"
        report += "=" * 60 + "\n"
        
        # Save report
        with open(f"{config.MODEL_PATH}training_report.txt", 'w') as f:
            f.write(report)
        
        print(report)

# =============================================
# MAIN EXECUTION
# =============================================

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Stocky AI models')
    parser.add_argument('--symbols', nargs='+', help='List of symbols to train on')
    parser.add_argument('--optimize', action='store_true', help='Use Optuna for hyperparameter optimization')
    parser.add_argument('--backtest-only', action='store_true', help='Run backtesting only')
    
    args = parser.parse_args()
    
    # Run training pipeline
    pipeline = TrainingPipeline()
    
    if args.backtest_only:
        logger.info("Running backtesting only...")
        # Load existing models and run backtesting
        if os.path.exists(f"{config.MODEL_PATH}ensemble_model.pkl"):
            ensemble_data = joblib.load(f"{config.MODEL_PATH}ensemble_model.pkl")
            logger.info("Models loaded successfully")
            logger.info(f"Last trained: {ensemble_data['trained_date']}")
        else:
            logger.error("No trained models found. Please run training first.")
    else:
        # Run full training pipeline
        ensemble_data = pipeline.run(symbols=args.symbols)
        
        logger.info("Training completed successfully!")
        logger.info(f"Models saved to: {config.MODEL_PATH}")
        logger.info("You can now start the FastAPI backend to serve predictions.")