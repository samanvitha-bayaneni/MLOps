from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd

app = FastAPI(
    title="Your Model API",
    description="API for making predictions with your MLflow model.",
    version="0.1.0",
)


# Define the request body structure — update this to match your model inputs

from pydantic import BaseModel, Field

class PredictionRequest(BaseModel):
    Year: int
    Economic_Loss_Million: float = Field(..., alias="Economic Loss (Million $)")
    Avg_Waste_per_Capita_Kg: float = Field(..., alias="Avg Waste per Capita (Kg)")
    Population_Million: float = Field(..., alias="Population (Million)")
    Household_Waste_Percent: float = Field(..., alias="Household Waste (%)")

    class Config:
        allow_population_by_field_name = True



# Global variable to hold the loaded model
model = None


# Load the MLflow model on startup
@app.on_event("startup")
async def startup_event():
    global model
    model_path = "/Users/samanvitha/MLOps/notebooks/mlartifacts/2/38d6e3f1b8014c51b0e37b857c3b6b98/artifacts/model"
    model = mlflow.pyfunc.load_model(model_path)


# Prediction endpoint
@app.post("/predict")
async def predict(request_data: PredictionRequest):
    try:
        input_df = pd.DataFrame([request_data.dict(by_alias=True)])
        prediction = model.predict(input_df)
        return {"prediction": prediction.tolist()}
    except Exception as e:
        return {"error": str(e)}



# Root endpoint
@app.get("/")
async def root():
    return {"message": "Your Model API is online"}


# Run the app locally
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
