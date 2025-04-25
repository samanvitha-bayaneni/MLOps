FROM python:3.13-slim

# Upgrade pip
RUN pip install -U pip

# Set working directory
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .

# Install dependencies
RUN pip install -r requirements.txt

# Copy the rest of the code
COPY reddit_model_pipeline.joblib redditApp.py ./

# Expose port (FastAPI default if you don't override it)
EXPOSE 8000

# Start the app
ENTRYPOINT ["uvicorn", "redditApp:app", "--host", "0.0.0.0", "--port", "8000"]
