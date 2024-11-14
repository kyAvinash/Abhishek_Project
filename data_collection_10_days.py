import random
import csv
from datetime import datetime, timedelta

# Function to simulate random sensor data for a day
def generate_random_data_for_day(date):
    data = []
    for hour in range(24):  # Assuming hourly data for each day
        # Simulate random sensor data
        water_level = random.randint(0, 1023)  # Water level sensor (0-1023 range)
        moisture = random.randint(0, 1023)     # Soil moisture sensor (0-1023 range)
        temperature = random.uniform(15, 40)   # Temperature sensor (15-40 Celsius)
        gas_level = random.randint(0, 1023)    # Gas sensor (0-1023 range)
        
        # Define status based on water level
        if water_level < 20:
            water_status = "Empty"
        elif water_level < 350:
            water_status = "Low"
        elif water_level < 510:
            water_status = "Medium"
        else:
            water_status = "High"

        # Define status based on soil moisture
        if moisture > 950:
            moisture_status = "Dry"
        elif moisture >= 400 and moisture <= 950:
            moisture_status = "Medium"
        else:
            moisture_status = "Wet"

        # Define status based on gas detection
        if gas_level > 700:
            gas_status = "Smoke"
        elif gas_level > 300:
            gas_status = "CO"
        elif gas_level > 500:
            gas_status = "LPG"
        else:
            gas_status = "Safe"

        # Prepare the timestamp for each reading
        timestamp = date + timedelta(hours=hour)

        # Collect all sensor data and their status with timestamp
        data.append([timestamp, water_level, moisture, temperature, gas_level, water_status, moisture_status, gas_status])

    return data

# Function to generate data for the last 15 days
def generate_data_for_last_15_days():
    data = []
    today = datetime.today()  # Current date
    start_date = today - timedelta(days=15)  # Start 15 days ago
    
    for day in range(15):  # 15 days of data
        current_date = start_date + timedelta(days=day)
        day_data = generate_random_data_for_day(current_date)
        data.extend(day_data)  # Add the day's data to the overall data list
    
    return data

# Generate the data for the last 15 days
data = generate_data_for_last_15_days()

# Writing to CSV file
with open('sensor_data_last_15_days.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Timestamp', 'WaterLevel', 'SoilMoisture', 'Temperature', 'GasLevel', 'WaterStatus', 'MoistureStatus', 'GasStatus'])
    writer.writerows(data)

print("CSV file 'sensor_data_last_15_days.csv' has been created with 15 days of sample data.")
