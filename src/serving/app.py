from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import logging

# Initialize FastAPI app
app = FastAPI(title="Remote Work Health Prediction API")

# Load serialized model and preprocessing objects
try:
    model = joblib.load("models/model.joblib")
    label_encoders = joblib.load("models/label_encoders.joblib")
    scaler = joblib.load("models/scaler.joblib")
    mental_imputer = joblib.load("models/mental_imputer.joblib")
except Exception as e:
    logging.error(f"Error loading model or preprocessing objects: {e}")
    raise

# Define input data model
class HealthInput(BaseModel):
    Gender: str
    Region: str
    Industry: str
    Job_Role: str
    Work_Arrangement: str
    Mental_Health_Status: str
    Physical_Health_Issues: str
    Salary_Range: str
    Age: float
    Hours_Per_Week: float
    Work_Life_Balance_Score: float
    Social_Isolation_Score: float

# Define prediction endpoint
@app.post("/predict")
async def predict(input_data: HealthInput):
    try:
        # Convert input to DataFrame
        data = pd.DataFrame([input_data.dict()])

        # Preprocessing
        # Impute Mental_Health_Status
        data['Mental_Health_Status'] = mental_imputer.transform(data[['Mental_Health_Status']]).ravel()
        # Impute Physical_Health_Issues
        data['Physical_Health_Issues'] = data['Physical_Health_Issues'].fillna('None')
        # Encode categorical variables
        categorical_cols = ['Gender', 'Region', 'Industry', 'Job_Role', 'Work_Arrangement',
                           'Mental_Health_Status', 'Physical_Health_Issues', 'Salary_Range']
        for col in categorical_cols:
            if col in label_encoders:
                try:
                    data[col] = label_encoders[col].transform(data[col])
                except ValueError as e:
                    valid_labels = label_encoders[col].classes_.tolist()
                    raise ValueError(f"Invalid value for {col}: {data[col].iloc[0]}. Valid values are: {valid_labels}")
            else:
                raise ValueError(f"No label encoder for column {col}")
        # Scale numerical features
        numerical_cols = ['Age', 'Hours_Per_Week', 'Work_Life_Balance_Score', 'Social_Isolation_Score']
        data[numerical_cols] = scaler.transform(data[numerical_cols])

        # Feature Engineering
        data['Work_Load'] = data['Hours_Per_Week'] * data['Social_Isolation_Score']
        data['Health_Stress_Index'] = data['Work_Life_Balance_Score'] * data['Mental_Health_Status']
        data['Age_Group'] = pd.cut(data['Age'], bins=[20, 30, 40, 50, 60, 70], labels=[0, 1, 2, 3, 4], include_lowest=True)
        data['Age_Group'] = data['Age_Group'].cat.codes

        # Prepare feature matrix
        X = data[categorical_cols + numerical_cols + ['Work_Load', 'Health_Stress_Index', 'Age_Group']]

        # Make prediction
        prediction = model.predict(X)[0]
        return {"Burnout_Level": int(prediction)}
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))