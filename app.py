import pandas as pd
import joblib
from flask import Flask, request, jsonify
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the trained model and label encoder
model = joblib.load('water_status_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Initialize Flask app
app = Flask(__name__)

# Function to prepare the input data for prediction
def prepare_input_data(data):
    input_data = {
        'WaterLevel': data['WaterLevel'],
        'SoilMoisture': data['SoilMoisture'],
        'Temperature': data['Temperature'],
        'GasLevel': data['GasLevel']
    }
    input_df = pd.DataFrame([input_data])
    return input_df

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data from the POST request (assumes JSON format)
        data = request.get_json()

        # Prepare the data for prediction
        input_df = prepare_input_data(data)

        # Make prediction using the trained model
        prediction = model.predict(input_df)

        # Decode the predicted label (WaterStatus)
        predicted_label = label_encoder.inverse_transform(prediction)

        # Return the predicted result
        return jsonify({'predicted_water_status': predicted_label[0]})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
