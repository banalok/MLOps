from fastapi import FastAPI
from sklearn.pipeline import Pipeline
import uvicorn
from data_types import PredictionDataset
import pandas as pd
import joblib
from pathlib import Path

app = FastAPI()
current_dir = Path(__file__).parent
model_dir = current_dir/"models"/"models"/"xgb.joblib"
preprocessor_path = model_dir.parent.parent/"transformers"/"preprocessor.joblib"
output_transformer_path = preprocessor_path.parent/"output_transformer.joblib"

model = joblib.load(model_dir)
preprocessor = joblib.load(preprocessor_path)
output_transformer = joblib.load(output_transformer_path)

pipeline = Pipeline(steps=[('preprocessing', preprocessor), ('regressor', model)])

@app.get('/')
def home():
    return "This is the Homepage"

@app.post('/predict')
def predict_duration(test_data:PredictionDataset):
    X_test = pd.DataFrame(
        data = {
            'vendor_id':test_data.vendor_id,
            'passenger_count':test_data.passenger_count,
            'pickup_longitude':test_data.pickup_longitude,
            'pickup_latitude':test_data.pickup_latitude,
            'dropoff_longitude':test_data.dropoff_longitude,
            'dropoff_latitude':test_data.dropoff_latitude,
            'pickup_hour':test_data.pickup_hour,
            'pickup_date':test_data.pickup_date,
            'pickup_month':test_data.pickup_month,
            'pickup_day':test_data.pickup_day,
            'is_weekend':test_data.is_weekend,
            'haversine_distance':test_data.haversine_distance,
            'euclidean_distance':test_data.euclidean_distance,
            'manhattan_distance':test_data.manhattan_distance
         }, index=[0]
    )

    predictions = pipeline.predict(X_test).reshape(-1,1)

    output_inverse_transformed = output_transformer.inverse_transform(predictions)

    return f"The predicted trip duration is {output_inverse_transformed[0][0]: .2f} minutes"

if __name__ == "__main__":
    uvicorn.run(app="app:app", port=8000)