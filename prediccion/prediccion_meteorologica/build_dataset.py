import pandas as pd

df_cloudiness = pd.read_csv("prediccion_meteorologica/data/cloudiness.csv")   # (cloudiness_id, cloudiness)
df_dates = pd.read_csv("prediccion_meteorologica/data/dates.csv")             # (date_id, date)
df_observations = pd.read_csv("prediccion_meteorologica/data/observations.csv")
df_seasons = pd.read_csv("prediccion_meteorologica/data/seasons.csv")         # (estacion_id, estacion)
df_weather = pd.read_csv("prediccion_meteorologica/data/weather.csv")         # (weather_id, weather)

# Interpolamos los valores numéricos en df_observations
df_observations['precipitation'].interpolate(method='linear', inplace=True)
df_observations['temp_max'].interpolate(method='linear', inplace=True)
df_observations['temp_min'].interpolate(method='linear', inplace=True)
df_observations['wind'].interpolate(method='linear', inplace=True)

# Redondeamos a 2 decimales
df_observations = df_observations.round(2)

# Hacemos el merge de las diferentes tablas
df_merged = pd.merge(df_observations, df_dates, on='date_id', how='left')
df_merged = pd.merge(df_merged, df_seasons, on='estacion_id', how='left')
df_merged = pd.merge(df_merged, df_weather, on='weather_id', how='left')
df_merged = pd.merge(df_merged, df_cloudiness, on='cloudiness_id', how='left')

# Convertimos la columna 'date' a tipo datetime
df_merged['date'] = pd.to_datetime(df_merged['date'], errors='coerce')

# Creamos las nuevas columnas de día, mes y año
df_merged['day'] = df_merged['date'].dt.day
df_merged['month'] = df_merged['date'].dt.month
df_merged['year'] = df_merged['date'].dt.year

# Exportamos a CSV el resultado final
df_merged.to_csv("prediccion_meteorologica/data/final_dataset.csv", index=False)

print("Se generó final_dataset.csv con las tablas unificadas y las columnas de day, month y year.")
