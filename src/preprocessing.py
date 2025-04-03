import pandas as pd

# Load datasets
train_data = pd.read_csv("data/adult.data", header=None)
test_data = pd.read_csv("data/adult.test", header=None)

# Example preprocessing: Rename columns
columns = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
           "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
           "hours-per-week", "native-country", "income"]
train_data.columns = columns
test_data.columns = columns

# Save processed data
train_data.to_csv("data/processed_train_data.csv", index=False)
test_data.to_csv("data/processed_test_data.csv", index=False)