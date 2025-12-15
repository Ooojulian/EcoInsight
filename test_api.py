import requests
import random

# CONFIGURACIÓN
URL = "http://127.0.0.1:8000/predict"
N_FEATURES = 79  # El número que te pidió el error (Cámbialo si el error dice otro)
HORAS = 24       # La ventana de tiempo obligatoria

print(f"⚡ Generando datos simulados para {N_FEATURES} variables durante {HORAS} horas...")

# Generamos una matriz de (24 filas x 79 columnas) con números aleatorios entre 0 y 1
# Esto simula los datos normalizados que recibiría el servidor
datos_fake = [[random.random() for _ in range(N_FEATURES)] for _ in range(HORAS)]

payload = {
    "data": datos_fake
}

try:
    print("📡 Enviando petición POST a la API local...")
    response = requests.post(URL, json=payload)
    
    if response.status_code == 200:
        print("\n✅ ¡ÉXITO! La API respondió correctamente:")
        print(response.json())
    else:
        print(f"\n❌ Error {response.status_code}:")
        print(response.text)

except Exception as e:
    print(f"\n❌ Error de conexión: {e}")
    print("Asegúrate de que 'uvicorn main:app --reload' esté corriendo en otra terminal.")
