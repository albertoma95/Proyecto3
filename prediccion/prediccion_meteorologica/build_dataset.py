import pandas as pd

# 1) Leemos cada CSV
df_cloudiness = pd.read_csv("prediccion_meteorologica/data/cloudiness.csv")   # (cloudiness_id, cloudiness)
df_dates = pd.read_csv("prediccion_meteorologica/data/dates.csv")             # (date_id, date)
df_observations = pd.read_csv("prediccion_meteorologica/data/observations.csv")
df_seasons = pd.read_csv("prediccion_meteorologica/data/seasons.csv")         # (estacion_id, estacion)
df_weather = pd.read_csv("prediccion_meteorologica/data/weather.csv")         # (weather_id, weather)

df_observations['precipitation'].interpolate(method='linear', inplace=True)
df_observations['temp_max'].interpolate(method='linear', inplace=True)
df_observations['temp_min'].interpolate(method='linear', inplace=True)
df_observations['wind'].interpolate(method='linear', inplace=True)

df_merged = pd.merge(df_observations, df_dates, on='date_id', how='left')

df_merged = pd.merge(df_merged, df_seasons, on='estacion_id', how='left')

df_merged = pd.merge(df_merged, df_weather, on='weather_id', how='left')

df_merged = pd.merge(df_merged, df_cloudiness, on='cloudiness_id', how='left')

df_merged.to_csv("prediccion_meteorologica/data/final_dataset.csv", index=False)

print("Se generó final_dataset.csv con las tablas unificadas.")
