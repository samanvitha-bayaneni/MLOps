# Path configurations
DATA_PATH = '/Users/samanvitha/MLOps/data/global_food_wastage_dataset.csv'

# MLFlow settings
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
MLFLOW_EXPERIMENT_NAME = "Food Waste Prediction"
REGISTERED_MODEL_NAME = "best_food_waste_model"

# Feature columns (will be populated during training)
FEATURE_COLUMNS_PATH = '/Users/samanvitha/MLOps/data/feature_columns.pkl'
SCALER_PATH = '/Users/samanvitha/MLOps/data/scaler.pkl'