import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import random
from datetime import datetime, timedelta

# Generate random data for the last 15 days


def generate_data():
    data = []
    labels = ['Low', 'Medium', 'High']
    moisture_status = ['Dry', 'Moist', 'Wet']
    gas_status = ['No Gas', 'Low Gas', 'High Gas']

    # Generate random data for 15 days
    for i in range(15):
        date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
        water_level = random.randint(100, 1000)  # Random water level (in mm)
        # Random soil moisture (in percentage)
        soil_moisture = random.randint(100, 1000)
        # Random temperature (in degree Celsius)
        temperature = random.randint(15, 40)
        gas_level = random.randint(0, 1000)  # Random gas level (in ppm)
        # Random water status (Low, Medium, High)
        water_status = random.choice(labels)
        moisture_status_value = random.choice(
            moisture_status)  # Random moisture status
        gas_status_value = random.choice(gas_status)  # Random gas status

        data.append([date, water_level, soil_moisture, temperature,
                    gas_level, water_status, moisture_status_value, gas_status_value])

    return pd.DataFrame(data, columns=['Date', 'WaterLevel', 'SoilMoisture', 'Temperature', 'GasLevel', 'WaterStatus', 'MoistureStatus', 'GasStatus'])


# Generate the data
data = generate_data()

# Preprocess the data
label_encoder = LabelEncoder()

# Encoding categorical values
data['WaterStatus'] = label_encoder.fit_transform(data['WaterStatus'])
data['MoistureStatus'] = label_encoder.fit_transform(data['MoistureStatus'])
data['GasStatus'] = label_encoder.fit_transform(data['GasStatus'])

# Features and Labels
X = data[['WaterLevel', 'SoilMoisture', 'Temperature', 'GasLevel']]  # Features
y = data['WaterStatus']  # Target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'water_status_model.pkl')

# Save the label encoder (for later use in the Flask app)
joblib.dump(label_encoder, 'label_encoder.pkl')

print("Model training complete and saved as 'water_status_model.pkl'")
