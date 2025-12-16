# 1. Usar una imagen base de Python oficial y ligera (Linux)
FROM python:3.11-slim

# 2. Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# 3. Copiar primero los requisitos (para aprovechar la caché de Docker)
COPY requirements.txt .

# 4. Instalar las dependencias del sistema necesarias
# (Evita problemas de caché y limpia después para que la imagen pese menos)
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copiar el resto del código (Modelos, API, Dashboard)
COPY . .

# 6. Exponer los puertos necesarios
# 8000 = Backend (FastAPI)
# 8501 = Frontend (Streamlit)
EXPOSE 8000
EXPOSE 8501

# 7. (Opcional) Comando por defecto (lo sobrescribiremos con Compose)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

