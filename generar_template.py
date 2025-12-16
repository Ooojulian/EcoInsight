import pandas as pd
import joblib
import numpy as np

# 1. Cargar el scaler para robarle los nombres de las columnas
scaler = joblib.load("scaler_maestro.pkl")

try:
    cols = scaler.feature_names_in_
except AttributeError:
    # Si la versión de scikit-learn es vieja, generamos nombres genéricos
    cols = [f"var_{i}" for i in range(79)]

# 2. Crear un DataFrame con datos de ejemplo (24 horas)
# Usamos datos aleatorios realistas para rellenar
data = np.random.uniform(10, 50, size=(24, len(cols)))
df_template = pd.DataFrame(data, columns=cols)

# 3. Guardar en la carpeta del proyecto
df_template.to_csv("datos_ejemplo.csv", index=False)
print("✅ Archivo 'datos_ejemplo.csv' generado con éxito.")
