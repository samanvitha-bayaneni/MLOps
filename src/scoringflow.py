from metaflow import FlowSpec, step, Parameter, JSONType
import pandas as pd
import mlflow
import numpy as np
import pickle
from feature_engineering import preprocess_data
from config import *

class ScoringFlow(FlowSpec):
    input_data = Parameter('input_data', type=JSONType, required=True)

    @step
    def start(self):
        # Load model and preprocessing artifacts
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        self.model = mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL_NAME}/latest")
        
        # Load feature columns and scaler
        try:
            with open(FEATURE_COLUMNS_PATH, 'rb') as f:
                self.feature_columns = pickle.load(f)
            with open(SCALER_PATH, 'rb') as f:
                self.scaler = pickle.load(f)
        except Exception as e:
            print(f"Error loading preprocessing artifacts: {e}")
            raise
            
        # Create DataFrame from input
        self.raw_df = pd.DataFrame([self.input_data])
        self.next(self.preprocess)

    @step
    def preprocess(self):
        # Preprocess the input
        df = preprocess_data(self.raw_df)

        # Optional: If this was used during training
        if 'Total Waste (Tons)' in df.columns and 'Population (Million)' in df.columns:
            df['Avg Waste per Capita (Kg)'] = (
                df['Total Waste (Tons)'] * 1000 / df['Population (Million)']
            )

        # Drop the target if it's in the input
        df.drop(columns=['Economic Loss (Million $)'], errors='ignore', inplace=True)

        # One-hot encoding for categorical variables
        for col in ['Country', 'Food Category']:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col)
                df = pd.concat([df.drop(columns=[col]), dummies], axis=1)

        # Ensure all expected features are present
        expected_features = self.scaler.feature_names_in_
        for col in expected_features:
            if col not in df.columns:
                df[col] = 0

        # Drop any extra columns not seen during training
        df = df[expected_features]

        # Apply scaler
        # Get expected model input columns
        expected_features = self.feature_columns

        # Add missing features
        for col in expected_features:
            if col not in df.columns:
                df[col] = 0

        # Drop extra columns
        df = df[expected_features]

        # Scale only numeric columns
        # Get numeric columns that scaler expects
        expected_numeric = self.scaler.feature_names_in_

        # Ensure all expected numeric columns exist in the current df
        for col in expected_numeric:
            if col not in df.columns:
                df[col] = 0

        # Align numeric cols and transform
        df[expected_numeric] = self.scaler.transform(df[expected_numeric])

        df.drop(columns=["Economic Loss (Million $)"], errors="ignore", inplace=True)
        # Save final input DataFrame
        self.df = df



        self.next(self.predict)



    @step
    def predict(self):
        self.prediction = self.model.predict(self.df)
        self.next(self.end)

    @step
    def end(self):
        print(f"Predicted Economic Loss: ${self.prediction[0]:.2f} million")

if __name__ == '__main__':
    ScoringFlow()