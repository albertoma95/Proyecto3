import pandas as pd

# 1) Leemos cada CSV
df_cloudiness = pd.read_csv("prediccion_meteorologica/data/cloudiness.csv")   # (cloudiness_id, cloudiness)
df_dates = pd.read_csv("prediccion_meteorologica/data/dates.csv")             # (date_id, date)
df_observations = pd.read_csv("prediccion_meteorologica/data/observations.csv")
df_seasons = pd.read_csv("prediccion_meteorologica/data/seasons.csv")         # (estacion_id, estacion)
df_weather = pd.read_csv("prediccion_meteorologica/data/weather.csv")         # (weather_id, weather)

# Unimos las observaciones con las fechas
df_merged = pd.merge(df_observations, df_dates, on='date_id', how='left')

# Ordenar por la columna 'date'
df_merged = df_merged.sort_values(by='date').reset_index(drop=True)

# Realizar la interpolación
df_merged['precipitation'].interpolate(method='linear', inplace=True)
df_merged['temp_max'].interpolate(method='linear', inplace=True)
df_merged['temp_min'].interpolate(method='linear', inplace=True)
df_merged['wind'].interpolate(method='linear', inplace=True)

# Redondear valores
df_merged = df_merged.round(2)

# Mergear con las otras tablas
df_merged = pd.merge(df_merged, df_seasons, on='estacion_id', how='left')
df_merged = pd.merge(df_merged, df_weather, on='weather_id', how='left')
df_merged = pd.merge(df_merged, df_cloudiness, on='cloudiness_id', how='left')

# Convertimos la columna 'date' a tipo datetime
df_merged['date'] = pd.to_datetime(df_merged['date'], errors='coerce')

# Creamos las columnas día, mes y año
df_merged['day'] = df_merged['date'].dt.day
df_merged['month'] = df_merged['date'].dt.month
df_merged['year'] = df_merged['date'].dt.year

# (Opcional) Si no quieres conservar la columna 'date' original, la puedes eliminar:
# df_merged.drop(columns=['date'], inplace=True)

# Exportamos el DataFrame final a CSV
df_merged.to_csv("prediccion_meteorologica/data/final_dataset.csv", index=False)

print("Se generó final_dataset.csv con las tablas unificadas y columnas de día, mes y año.")
