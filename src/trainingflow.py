from metaflow import FlowSpec, step, Parameter
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import numpy as np
from feature_engineering import preprocess_data, normalize_features, get_feature_columns
from config import *
import pickle

class TrainingFlow(FlowSpec):
    test_size = Parameter('test_size', default=0.2, type=float)
    random_state = Parameter('random_state', default=42, type=int)
    
    @step
    def start(self):
        # Load and preprocess data
        raw_df = pd.read_csv(DATA_PATH)
        self.df = preprocess_data(raw_df)
        
        # Identify numeric columns
        numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.df, self.scaler = normalize_features(self.df, numeric_cols)
        
        # Store feature columns for inference
        global FEATURE_COLUMNS
        FEATURE_COLUMNS = get_feature_columns(self.df)

        with open(FEATURE_COLUMNS_PATH, 'wb') as f:
            pickle.dump(FEATURE_COLUMNS, f)
        with open(SCALER_PATH, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Split data
        X = self.df.drop(columns=['Economic Loss (Million $)'])
        y = self.df['Economic Loss (Million $)']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        self.next(self.train_rf, self.train_svr)
    
    @step
    def train_rf(self):
        self.model_name = "Random Forest"
        self.model = RandomForestRegressor(random_state=self.random_state)
        self.model.fit(self.X_train, self.y_train)
        
        # Evaluate the model
        self.y_pred = self.model.predict(self.X_test)
        self.mse = mean_squared_error(self.y_test, self.y_pred)
        self.rmse = np.sqrt(self.mse)
        self.r2 = r2_score(self.y_test, self.y_pred)
        
        self.next(self.choose_model)
    
    @step
    def train_svr(self):
        self.model_name = "SVR"
        self.model = SVR(kernel='rbf')
        self.model.fit(self.X_train, self.y_train)
        
        # Evaluate the model
        self.y_pred = self.model.predict(self.X_test)
        self.mse = mean_squared_error(self.y_test, self.y_pred)
        self.rmse = np.sqrt(self.mse)
        self.r2 = r2_score(self.y_test, self.y_pred)
        
        self.next(self.choose_model)
    
    @step
    def choose_model(self, inputs):
        # For regression, we want to minimize RMSE or maximize R²
        # Let's use R² as our selection criterion (higher is better)
        self.best_r2 = -float('inf')
        for inp in inputs:
            if inp.r2 > self.best_r2:
                self.best_r2 = inp.r2
                self.best_model = inp.model
                self.best_model_name = inp.model_name
                self.best_rmse = inp.rmse
                self.best_mse = inp.mse
        
        # Log to MLFlow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        with mlflow.start_run():
            mlflow.log_params({
                "test_size": self.test_size,
                "random_state": self.random_state,
                "model_type": self.best_model_name
            })
            mlflow.log_metrics({
                "r2_score": self.best_r2,
                "rmse": self.best_rmse,
                "mse": self.best_mse
            })
            mlflow.sklearn.log_model(
                self.best_model, 
                "model", 
                registered_model_name=REGISTERED_MODEL_NAME
            )
        
        self.next(self.end)
    
    @step
    def end(self):
        print(f"Best model: {self.best_model_name}")
        print(f"R² Score: {self.best_r2:.4f}")
        print(f"RMSE: {self.best_rmse:.4f}")

if __name__ == '__main__':
    TrainingFlow()