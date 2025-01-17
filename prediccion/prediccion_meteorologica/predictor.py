import pandas as pd
import joblib
import numpy as np
from sklearn.feature_selection import chi2
import json
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFE

# ------------------------------------------------------------------------------
# OPCIÓN ELEGIDA POR EL USUARIO
# ------------------------------------------------------------------------------

print("Seleccione la opción:")
print("1) Entrenar TODOS los modelos (búsqueda con diferentes CV y métricas).")
print("2) Entrenar SOLO el modelo óptimo (f1_macro, CV=10, C=100, gamma=0.1, rbf).")
opcion = input("Ingrese 1 o 2: ")

# ------------------------------------------------------------------------------
# 0) CARGA Y PREPROCESAMIENTO DEL DATASET
# ------------------------------------------------------------------------------
df = pd.read_csv("prediccion_meteorologica/data/final_dataset.csv")

# Procesamiento de fechas
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day

X = df[['precipitation','wind','visibility','humidity']]
y = df['weather_id']

# Escalado
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Balanceo con SMOTE
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_scaled, y)

# Selección de características con RFE
rfe = RFE(
    estimator=RandomForestClassifier(random_state=42),
    n_features_to_select=10
)
rfe.fit(X_resampled, y_resampled)
X_rfe = rfe.transform(X_resampled)

selected_features = [col for col, sel in zip(X.columns, rfe.support_) if sel]
print("Características seleccionadas con RFE:", selected_features)

# División en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X_rfe, y_resampled, test_size=0.2, random_state=42
)

# (Opcional) Análisis de importancia de características
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
feature_importances = rf_model.feature_importances_
for feat, imp in zip(selected_features, feature_importances):
    print(f"{feat}: {imp:.4f}")


if opcion == '1':
    # ------------------------------------------------------------------------------
    # 1) ENTRENAR TODOS LOS MODELOS CON DIFERENTES CV Y MÉTRICAS
    # ------------------------------------------------------------------------------
    cv_values = [5, 10]
    scorings = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro', 'roc_auc_ovr']
    results = {}

    param_grid = [
        {'kernel': ['linear'], 'C': [1, 10, 100], 'class_weight': ['balanced']},
        {'kernel': ['poly'], 'C': [1, 10, 100], 'gamma': [0.1, 0.01], 'class_weight': ['balanced']},
        {'kernel': ['rbf'], 'C': [1, 10, 100], 'gamma': [0.1, 0.01, 0.001], 'class_weight': ['balanced']},
        {'kernel': ['sigmoid'], 'C': [1, 10, 100], 'gamma': [0.1, 0.01], 'class_weight': ['balanced']}
    ]

    best_model = None

    for cv in cv_values:
        for scoring in scorings:
            print(f"\nOptimización con CV = {cv} y Scoring = {scoring}")
            grid_search = GridSearchCV(
                SVC(probability=True),
                param_grid,
                cv=cv,
                scoring=scoring,
                verbose=2
            )
            grid_search.fit(X_train, y_train)
            results[(cv, scoring)] = grid_search.best_params_

            print(f"Mejores parámetros encontrados con CV={cv} y Scoring={scoring}: {grid_search.best_params_}")

            # Entrenamiento del mejor modelo
            model = grid_search.best_estimator_
            model.fit(X_train, y_train)

            # Predicción en el conjunto de prueba
            y_pred = model.predict(X_test)

            # Informe de clasificación
            print(f"Informe de Clasificación para CV={cv} y Scoring={scoring}:")
            print(classification_report(y_test, y_pred))

            # Precisión en entrenamiento y prueba
            y_train_pred = model.predict(X_train)
            train_accuracy = accuracy_score(y_train, y_train_pred) * 100
            test_accuracy = accuracy_score(y_test, y_pred) * 100

            print(f"Precisión en entrenamiento para CV={cv} y Scoring={scoring}: {train_accuracy:.2f}%")
            print(f"Precisión en prueba para CV={cv} y Scoring={scoring}: {test_accuracy:.2f}%")
            cm = confusion_matrix(y_test, y_pred)
            print(f"Matriz de Confusión para CV={cv} y Scoring={scoring}:\n{cm}")

            # Guardamos el último 'model' como mejor modelo,
            # aunque en la práctica podrías guardar condicionalmente
            # el modelo con mejor métrica
            best_model = model

    # Al final guardamos el último (o el que tú definas como mejor)
    if best_model is not None:
        joblib.dump(best_model, "prediccion_meteorologica/models/svm_model.pkl")
    joblib.dump(scaler, "prediccion_meteorologica/models/scaler.pkl")
    print("Modelo(s) entrenado(s) y guardado(s) correctamente.")

elif opcion == '2':
    # ------------------------------------------------------------------------------
    # 2) ENTRENAR SOLO EL MODELO ÓPTIMO CON PARÁMETROS DEFINIDOS
    # ------------------------------------------------------------------------------
    print("\nEntrenando SOLO el modelo óptimo con parámetros fijos:")
    print("Parámetros: f1_macro, CV=10, C=100, gamma=0.1, kernel='rbf', class_weight='balanced'")

    # Configuramos el modelo SVC con los parámetros deseados
    optimal_model = SVC(
        C=100,
        gamma=0.1,
        kernel='rbf',
        class_weight='balanced',
        probability=True
    )

    # Entrenamiento con validación cruzada de 10 pliegues
    # (Si deseas, puedes omitir GridSearchCV y entrenar directamente)


    # Entrenamos con el conjunto de entrenamiento completo
    optimal_model.fit(X_train, y_train)
    y_pred = optimal_model.predict(X_test)

    print("\nInforme de Clasificación (modelo óptimo):")
    print(classification_report(y_test, y_pred))

    train_accuracy = accuracy_score(y_train, optimal_model.predict(X_train)) * 100
    test_accuracy = accuracy_score(y_test, y_pred) * 100

    print(f"Precisión en entrenamiento: {train_accuracy:.2f}%")
    print(f"Precisión en prueba: {test_accuracy:.2f}%")
    cm = confusion_matrix(y_test, y_pred)
    print("Matriz de confusión (modelo óptimo):\n", cm)

    X_non_negative = X_scaled - X_scaled.min(axis=0)

    # Validar si todos los valores son no negativos
    if (X_non_negative < 0).any():
        raise ValueError("Los datos no son no negativos después de la transformación.")

    # Calcular los puntajes de Chi-cuadrado
    chi2_scores, p_values = chi2(X_non_negative, y)

    # Crear un diccionario con los resultados
    chi2_results = {feature: score for feature, score in zip(X.columns, chi2_scores)}

    print("\nPesos de las variables utilizando Chi-cuadrado:")
    for feature, score in chi2_results.items():
        print(f"{feature}: {score:.4f}")

    joblib.dump(optimal_model, "prediccion_meteorologica/models/svm_model.pkl")
    joblib.dump(scaler, "prediccion_meteorologica/models/scaler.pkl")
    print("Modelo óptimo entrenado y guardado correctamente.")
    # Convertir el informe de clasificación a un diccionario
    classification_report_dict = classification_report(y_test, y_pred, output_dict=True)

    # Guardar el informe de clasificación en JSON
    with open("prediccion_meteorologica/reports/classification_report.json", "w") as f:
        json.dump(classification_report_dict, f, indent=4)

    # Guardar la matriz de confusión en JSON
    confusion_matrix_dict = {
        "matrix": cm.tolist(),
        "labels": list(set(y_test))  # Etiquetas únicas
    }
    with open("prediccion_meteorologica/reports/confusion_matrix.json", "w") as f:
        json.dump(confusion_matrix_dict, f, indent=4)

    # Guardar las precisiones en JSON
    metrics = {
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy
    }
    with open("prediccion_meteorologica/reports/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("Resultados guardados en archivos JSON.")

else:
    print("Opción no válida. Por favor, seleccione 1 o 2.")
