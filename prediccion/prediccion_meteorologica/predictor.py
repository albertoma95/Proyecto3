import pandas as pd
import joblib
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Cargar el dataset
df = pd.read_csv("prediccion_meteorologica/data/final_dataset.csv")

# Procesamiento de fechas
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day

# Selección de características y variable objetivo
X = df[['precipitation', 'temp_max', 'temp_min', 'wind',
        'humidity', 'pressure', 'solar_radiation', 'visibility',
        'year', 'month', 'day']]
y = df['weather_id']

# Escalado de características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Balanceo de clases con SMOTE
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_scaled, y)

# División de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

# Análisis de importancia de características usando Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_resampled, y_resampled)

feature_importances = rf_model.feature_importances_
sorted_idx = np.argsort(feature_importances)[::-1]

print("Importancia de las características:")
for idx in sorted_idx:
    print(f"{X.columns[idx]}: {feature_importances[idx]:.4f}")

# Optimización de hiperparámetros con diferentes CV
cv_values = [3, 5, 10, 20]
results = {}

param_grid = [
    {'kernel': ['linear'], 'C': [1, 10, 100], 'class_weight': ['balanced']},
    {'kernel': ['poly'], 'C': [1, 10, 100], 'gamma': [0.1, 0.01], 'class_weight': ['balanced']},
    {'kernel': ['rbf'], 'C': [1, 10, 100], 'gamma': [0.1, 0.01, 0.001], 'class_weight': ['balanced']},
    {'kernel': ['sigmoid'], 'C': [1, 10, 100], 'gamma': [0.1, 0.01], 'class_weight': ['balanced']}
]

for cv in cv_values:
    print(f"Optimización con CV = {cv}")
    grid_search = GridSearchCV(SVC(probability=True), param_grid, cv=cv, scoring='f1_macro', verbose=2)
    grid_search.fit(X_train, y_train)
    results[cv] = grid_search.best_params_

    print(f"Mejores parámetros encontrados con CV={cv}: {grid_search.best_params_}")

    # Entrenamiento del mejor modelo
    model = grid_search.best_estimator_
    model.fit(X_train, y_train)

    # Predicción en el conjunto de prueba
    y_pred = model.predict(X_test)

    # Informe de clasificación
    print(f"Informe de Clasificación para CV={cv}:")
    print(classification_report(y_test, y_pred))

    # Precisión en entrenamiento y prueba
    y_train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred) * 100
    test_accuracy = accuracy_score(y_test, y_pred) * 100

    print(f"Precisión en entrenamiento para CV={cv}: {train_accuracy:.2f}%")
    print(f"Precisión en prueba para CV={cv}: {test_accuracy:.2f}%")

# Guardar el mejor modelo y el scaler
joblib.dump(model, "prediccion_meteorologica/models/svm_model.pkl")
joblib.dump(scaler, "prediccion_meteorologica/models/scaler.pkl")
