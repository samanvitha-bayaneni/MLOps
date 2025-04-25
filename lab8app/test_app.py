import requests
import json

# Define the URL of your FastAPI prediction endpoint
url = "http://127.0.0.1:8000/predict"

# Sample input matching the model's expected schema
sample_data = {
    "Year": 2022,
    "Economic Loss (Million $)": 123.45,
    "Avg Waste per Capita (Kg)": 0.67,
    "Population (Million)": 500.0,
    "Household Waste (%)": 65.4
}

try:
    # Send POST request to FastAPI
    response = requests.post(url, json=sample_data)
    response.raise_for_status()  # Raises HTTPError for bad responses

    # Parse and print the prediction
    prediction = response.json()
    print("Prediction:", prediction)

except requests.exceptions.RequestException as e:
    print("Request Error:", e)
except json.JSONDecodeError:
    print("Error: Could not decode JSON response.")
