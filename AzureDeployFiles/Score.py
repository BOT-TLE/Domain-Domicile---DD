import json
import joblib
import numpy as np
from azureml.core.model import Model

def init():
    global model
    model_path = Model.get_model_path('Housing_price_prediction')  
    model = joblib.load(model_path)

def run(raw_data):
    try:
        data = json.loads(raw_data)
        
        # Ensure 'data' is a list
        if not isinstance(data, list):
            raise ValueError("Input data must be a list of JSON objects.")
        
        # Convert JSON data to numpy array
        data_array = np.array(data)

        # Perform prediction using the loaded scikit-learn model
        result = model.predict(data_array)

        # You can return the result as a dictionary or in any desired format
        return json.dumps({"result": result.tolist()})
    except Exception as e:
        return json.dumps({"error": str(e)})