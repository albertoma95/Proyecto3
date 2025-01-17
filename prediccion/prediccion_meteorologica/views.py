import datetime
import json
import joblib
from django.shortcuts import render

def predict_weather(request):
    prediction = None
    weather_info = None

    # Variables para las métricas (inicialmente vacías)
    classification_report = {}
    confusion_matrix = {}
    metrics = {}

    if request.method == 'POST':
        data = request.POST


        # 3. Construir el vector de características
        #    *Nota*: Esto debe concordar con el orden de columnas que usaste al entrenar el modelo
        features = [
            float(data['precipitation']),
            float(data['wind']),
            float(data['visibility']),
            float(data['humidity']),
        ]
        print(features)

        # 4. Cargar modelo y scaler
        model = joblib.load("prediccion_meteorologica/models/svm_model.pkl")
        scaler = joblib.load("prediccion_meteorologica/models/scaler.pkl")

        # 5. Escalar las características y predecir
        features_scaled = scaler.transform([features])
        prediction = int(model.predict(features_scaled)[0])
        print(prediction)

        # 6. Diccionario para obtener la descripción del clima según la predicción
        weather_dict = {
            1: {
                'name': "Tormenta",
                'description': "Se esperan fuertes tormentas. ¡Precaución!",
                'image': "https://media.giphy.com/media/6ZhkSxi5KvORq/giphy.gif",
                'icon': "⚡"
            },
            2: {
                'name': "Lluvia",
                'description': "Día lluvioso, no olvides tu paraguas.",
                'image': "https://media.giphy.com/media/Ckt7qu9ksg5ByO2ibi/giphy.gif",
                'icon': "🌧"
            },
            3: {
                'name': "Nublado",
                'description': "Cielo cubierto y con pocas probabilidades de sol.",
                'image': "https://media.giphy.com/media/RO5XhlFWOPs6k/giphy.gif",
                'icon': "☁"
            },
            4: {
                'name': "Niebla",
                'description': "Visibilidad reducida por la niebla.",
                'image': "https://media.giphy.com/media/W0sgn9xy8Mul3ab0mG/giphy.gif",
                'icon': "🌫"
            },
            5: {
                'name': "Soleado",
                'description': "Día despejado y radiante. ¡A disfrutar!",
                'image': "https://media.giphy.com/media/bcJvDLgxVSPulkDYB4/giphy.gif",
                'icon': "☀"
            }
        }
        weather_info = weather_dict.get(prediction, None)

    # Lectura de archivos JSON con las métricas
    try:
        with open("prediccion_meteorologica/reports/classification_report.json", "r") as f:
            classification_report = json.load(f)
    except Exception as e:
        classification_report = {"error": "No se pudo cargar el classification report."}

    try:
        with open("prediccion_meteorologica/reports/confusion_matrix.json", "r") as f:
            confusion_matrix = json.load(f)
    except Exception as e:
        confusion_matrix = {"error": "No se pudo cargar la confusion matrix."}

    try:
        with open("prediccion_meteorologica/reports/metrics.json", "r") as f:
            metrics = json.load(f)
    except Exception as e:
        metrics = {"error": "No se pudo cargar las métricas."}

    context = {
        'prediction': prediction,
        'weather_info': weather_info,
        'classification_report': classification_report,
        'confusion_matrix': confusion_matrix,
        'metrics': metrics
    }

    return render(request, 'index.html', context)
