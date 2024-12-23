from django.shortcuts import render
from django.http import JsonResponse
import joblib
import numpy as np

def predict_weather(request):
    if request.method == 'POST':
        data = request.POST
        features = [
            float(data['precipitation']),
            float(data['temp_max']),
            float(data['temp_min']),
            float(data['wind']),
            float(data['humidity']),
            float(data['pressure']),
            float(data['solar_radiation']),
            float(data['visibility']),
        ]

        # Cargar el modelo y el escalador
        model = joblib.load("prediccion_meteorologica/models/svm_model.pkl")
        scaler = joblib.load("prediccion_meteorologica/models/scaler.pkl")

        features_scaled = scaler.transform([features])

        prediction = model.predict(features_scaled)[0]

        prediction = int(prediction)

        return JsonResponse({'prediction': prediction})

    return render(request, 'index.html')
