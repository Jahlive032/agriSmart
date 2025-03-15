from flask import Flask, render_template, request
from sklearn.linear_model import LogisticRegression
import numpy as np
import requests
import time

app = Flask(__name__)

# Clés API
OPENWEATHER_API_KEY = "4d59a8dfbe21daae109f9b455236b4aa"  # Remplace par ta clé OpenWeatherMap
AGRO_API_KEY = "2d2dee68d1290c7a8d4f1aa2adb61c50"      # Remplace par ta clé Agromonitoring

# Coordonnées des régions togolaises (lat, lon)
REGIONS = {
  "Kara": (9.5489, 1.1861),
  "Lome": (9.5489, 1.1861),
  "Atakpame": (9.5489, 1.1861),
  "Tchamba": (9.5489, 1.1861),
  "Dapaong": (9.5489, 1.1861),
}

# Modèle ML pour prédiction
X_train = np.array([[20, 25], [50, 28], [30, 30], [70, 26], [10, 32], [60, 27], [40, 33], [80, 24], [25, 35], [90, 22], [15, 29]])
y_train = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
model = LogisticRegression()
model.fit(X_train, y_train)

# Météo actuelle (OpenWeatherMap)
def get_weather(region):
    lat, lon = REGIONS[region]
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
    response = requests.get(url)
    return response.json()["main"]["temp"] if response.status_code == 200 else 27

# Prévisions 5 jours (OpenWeatherMap)
def get_forecast(region):
    lat, lon = REGIONS[region]
    url = f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        forecast = [f"J+{i+1}: {response.json()['list'][i*8]['main']['temp']}°C" for i in range(5)]
        return " | ".join(forecast)
    return "Prévisions indisponibles"

# Créer un polygone pour Agromonitoring
def create_polygon(region):
    lat, lon = REGIONS[region]
    url = f"http://api.agromonitoring.com/agro/1.0/polygons?appid={AGRO_API_KEY}"
    polygon = {
        "name": f"{region}_field_{int(time.time())}",
        "geo_json": {
            "type": "Feature",
            "properties": {},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [lon - 0.01, lat - 0.01],
                    [lon + 0.01, lat - 0.01],
                    [lon + 0.01, lat + 0.01],
                    [lon - 0.01, lat + 0.01],
                    [lon - 0.01, lat - 0.01]
                ]]
            }
        }
    }
    response = requests.post(url, json=polygon)
    if response.status_code == 201:
        print(f"Polygon created with ID: {response.json()['id']}")
        return response.json()["id"]
    else:
        print(f"Failed to create polygon: {response.status_code} - {response.text}")
        return None

# Récupérer NDVI (Agromonitoring)
def get_ndvi(poly_id):
    end = int(time.time())
    start = end - 86400 * 7  # Derniers 7 jours
    url = f"http://api.agromonitoring.com/agro/1.0/ndvi/history?polyid={poly_id}&start={start}&end={end}&appid={AGRO_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200 and response.json():
        return response.json()[-1]["ndvi"]  # Dernière valeur NDVI disponible
    return "Non disponible (données en attente)"
    # return 0.6  # Simulation pour l’atelier

# Simulation de Crop Map
def get_crop_map(region, crop, ndvi):
    return f"Champ à {region} : {crop} dominant, état estimé via NDVI ({ndvi})."

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    region = request.form['region']
    crop = request.form['crop']
    humidity = float(request.form['humidity'])

    # 1. Weather Data
    temp = get_weather(region)
    forecast = get_forecast(region)

    # 2. Satellite Imagery & Vegetation Indices
    poly_id = create_polygon(region)
    ndvi = get_ndvi(poly_id) if poly_id else "Non disponible"

    # 3. Prédiction
    input_data = np.array([[humidity, temp]])
    prediction = model.predict(input_data)[0]

    # Message personnalisé
    if prediction == 1:
        result = f"À {region} : Oui, plantez votre {crop} demain ! Humidité : {humidity}%, Température : {temp}°C."
    else:
        reason = "humidité trop basse" if humidity < 30 else "trop chaud" if temp > 35 else "conditions non optimales"
        result = f"À {region} : Non, attendez pour votre {crop}. {reason}. Humidité : {humidity}%, Température : {temp}°C."

    # 4. Crop Map & Analytics
    crop_map = get_crop_map(region, crop, ndvi)

    return render_template('index.html', prediction=result, ndvi=ndvi, forecast=forecast, crop_map=crop_map)

if __name__ == '__main__':
    app.run(debug=True)