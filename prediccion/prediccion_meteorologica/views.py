from django.shortcuts import render
import joblib

def predict_weather(request):
    prediction = None
    weather_info = None

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

        model = joblib.load("prediccion_meteorologica/models/svm_model.pkl")
        scaler = joblib.load("prediccion_meteorologica/models/scaler.pkl")

        features_scaled = scaler.transform([features])
        prediction = int(model.predict(features_scaled)[0])

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

    return render(request, 'index.html', {
        'prediction': prediction,
        'weather_info': weather_info
    })
