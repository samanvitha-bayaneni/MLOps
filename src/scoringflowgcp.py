from metaflow import FlowSpec, step, Parameter, JSONType, conda_base, retry, timeout, kubernetes
import pandas as pd
import mlflow
import numpy as np
import pickle
import logging
from feature_engineering import preprocess_data
import gcsfs

@conda_base(python="3.10", libraries={
    "pandas": "1.5.3",
    "scikit-learn": "1.2.2",
    "mlflow": "2.2.2",
    "numpy": "1.23.5",
    "gcsfs": "2023.6.0",
    "databricks-cli": "0.17.7"
})
class ScoringFlow(FlowSpec):
    input_data = Parameter('input_data', type=JSONType, required=True)

    @kubernetes(cpu=2, memory=4096)
    @retry(times=2)
    @timeout(seconds=300)
    @step
    def start(self):
        logging.basicConfig(level=logging.INFO)
        logging.info("Loading model and preprocessing artifacts...")

        mlflow.set_tracking_uri("https://mlflow-run-1043603561525.us-west2.run.app/")
        self.model = mlflow.sklearn.load_model(f"models:/food_waste_model/latest")

        try:
            fs = gcsfs.GCSFileSystem()
            with fs.open("gs://storage-tryagain9-metaflow-default/feature_columns.pkl", 'rb') as f:
                self.feature_columns = pickle.load(f)
            with fs.open("gs://storage-tryagain9-metaflow-default/scaler.pkl", 'rb') as f:
                self.scaler = pickle.load(f)
        except Exception as e:
            logging.error(f"Error loading preprocessing artifacts: {e}")
            raise


        self.raw_df = pd.DataFrame([self.input_data])
        self.next(self.preprocess)

    @kubernetes(cpu=2, memory=4096)
    @retry(times=2)
    @timeout(seconds=300)
    @step
    def preprocess(self):
        logging.info("Preprocessing input data...")

        df = preprocess_data(self.raw_df)

        if 'Total Waste (Tons)' in df.columns and 'Population (Million)' in df.columns:
            df['Avg Waste per Capita (Kg)'] = (
                df['Total Waste (Tons)'] * 1000 / df['Population (Million)']
            )

        df.drop(columns=['Economic Loss (Million $)'], errors='ignore', inplace=True)

        for col in ['Country', 'Food Category']:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col)
                df = pd.concat([df.drop(columns=[col]), dummies], axis=1)

        expected_features = self.feature_columns
        for col in expected_features:
            if col not in df.columns:
                df[col] = 0
        df = df[expected_features]

        numeric_cols = self.scaler.feature_names_in_
        for col in numeric_cols:
            if col not in df.columns:
                df[col] = 0

        df[numeric_cols] = self.scaler.transform(df[numeric_cols])

        self.df = df
        self.next(self.predict)

    @kubernetes(cpu=1, memory=2048)
    @retry(times=2)
    @timeout(seconds=180)
    @step
    def predict(self):
        logging.info("Running prediction...")
        self.prediction = self.model.predict(self.df)
        self.next(self.end)

    @step
    def end(self):
        print(f"\n🎯 Predicted Economic Loss: ${self.prediction[0]:.2f} million\n")


if __name__ == '__main__':
    ScoringFlow()
