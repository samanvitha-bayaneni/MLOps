from metaflow import FlowSpec, step, Parameter, retry, timeout, catch, conda_base, kubernetes, resources, schedule
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import numpy as np
import pickle
import logging
import gcsfs

from feature_engineering import preprocess_data, normalize_features, get_feature_columns
@schedule(hourly=True)
@conda_base(python="3.10", libraries={
    "pandas": "1.5.3",
    "scikit-learn": "1.2.2",
    "mlflow": "2.2.2",
    "numpy": "1.23.5",
    "imbalanced-learn": "0.10.1",
    "pip": "23.2.1",
    "setuptools": "65.6.3",
    "databricks-cli": "0.17.7",
    "gcsfs": "2023.6.0"
})
class TrainingFlow(FlowSpec):

    test_size = Parameter('test_size', default=0.2, type=float)
    random_state = Parameter('random_state', default=42, type=int)

    @resources(cpu=2, memory=4096)
    @timeout(seconds=300)
    @retry(times=2)
    @step
    def start(self):
        logging.basicConfig(level=logging.INFO)
        logging.info("Loading model and preprocessing artifacts...")

        mlflow.set_tracking_uri("https://mlflow-run-1043603561525.us-west2.run.app/")
        self.model = mlflow.sklearn.load_model(f"models:/food_waste_model/latest")

        try:
            fs = gcsfs.GCSFileSystem()
            
            # Check if files exist before loading
            path_feature_cols = "gs://storage-tryagain9-metaflow-default/feature_columns.pkl"
            path_scaler = "gs://storage-tryagain9-metaflow-default/scaler.pkl"
            
            if not fs.exists(path_feature_cols):
                raise FileNotFoundError(f"Feature columns file not found at {path_feature_cols}")
            
            if not fs.exists(path_scaler):
                raise FileNotFoundError(f"Scaler file not found at {path_scaler}")
                
            # Log file sizes
            feature_size = fs.info(path_feature_cols)['size']
            scaler_size = fs.info(path_scaler)['size']
            logging.info(f"Feature columns file size: {feature_size} bytes")
            logging.info(f"Scaler file size: {scaler_size} bytes")
            
            if feature_size == 0 or scaler_size == 0:
                raise ValueError("One or more preprocessing artifacts are empty files")
            
            # Use a try/except block for each file separately
            try:
                with fs.open(path_feature_cols, 'rb') as f:
                    self.feature_columns = pickle.load(f)
                logging.info(f"Successfully loaded feature columns: {len(self.feature_columns)} features")
            except Exception as e:
                logging.error(f"Error loading feature columns: {str(e)}")
                raise
                
            try:
                with fs.open(path_scaler, 'rb') as f:
                    self.scaler = pickle.load(f)
                logging.info("Successfully loaded scaler")
            except Exception as e:
                logging.error(f"Error loading scaler: {str(e)}")
                raise
                
        except Exception as e:
            logging.error(f"Error accessing preprocessing artifacts: {e}")
            raise

        self.raw_df = pd.DataFrame([self.input_data])
        self.next(self.preprocess)

    @kubernetes(cpu=4, memory=8192)
    @timeout(seconds=600)
    @retry(times=2)
    @catch(var="train_model_error")
    @step
    def train_model(self):
        model_type = self.input
        if model_type == 'rf':
            self.model_name = "Random Forest"
            self.model = RandomForestRegressor(random_state=self.random_state)
        else:
            self.model_name = "SVR"
            self.model = SVR(kernel='rbf')

        print(f"Training {self.model_name}...")
        self.model.fit(self.X_train, self.y_train)

        self.y_pred = self.model.predict(self.X_test)
        self.mse = mean_squared_error(self.y_test, self.y_pred)
        self.rmse = np.sqrt(self.mse)
        self.r2 = r2_score(self.y_test, self.y_pred)

        print(f"{self.model_name} Results -> R2: {self.r2:.4f}, RMSE: {self.rmse:.4f}")
        self.next(self.choose_model)

    @resources(cpu=2, memory=4096)
    @timeout(seconds=600)
    @retry(times=2)
    @catch(var="choose_model_error")
    @step
    def choose_model(self, inputs):
        self.best_r2 = -float('inf')
        for inp in inputs:
            if inp.r2 > self.best_r2:
                self.best_r2 = inp.r2
                self.best_model = inp.model
                self.best_model_name = inp.model_name
                self.best_rmse = inp.rmse
                self.best_mse = inp.mse

        mlflow.set_tracking_uri("https://mlflow-run-1043603561525.us-west2.run.app/")
        mlflow.set_experiment("metaflow-gcp-kubernetes")

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
                registered_model_name="food_waste_model"
            )
            print(f"MLflow Run ID: {mlflow.active_run().info.run_id}")

        self.next(self.end)

    @step
    def end(self):
        print(f"Best model: {self.best_model_name}")
        print(f"R² Score: {self.best_r2:.4f}")
        print(f"RMSE: {self.best_rmse:.4f}")


if __name__ == '__main__':
    TrainingFlow()
