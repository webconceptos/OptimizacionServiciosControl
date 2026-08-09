# Imagen de la API de la Parte 2 (Computacion Evolutiva).
# Escucha en el puerto 8001; el 8000 lo ocupa la API de la Parte 1 (tesis).
# Referencia: README.md
FROM python:3.11-slim

WORKDIR /app

# PYTHONUNBUFFERED: sin esto los logs quedan en el buffer y `docker logs` no
# muestra nada hasta que el proceso termina, lo que hace inservible el log
# estructurado de main.py.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Las dependencias van antes de copiar el codigo: asi la capa de pip se reutiliza
# de la cache mientras requirements.txt no cambie.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p outputs

EXPOSE 8001

# El puerto efectivo lo decide API_PORT del entorno (docker-compose lo inyecta con
# env_file). Si no hay .env ni variables, Params usa los valores por defecto
# documentados y la API queda en 8001.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import os,urllib.request,sys; \
url='http://127.0.0.1:%s/health' % os.getenv('API_PORT','8001'); \
sys.exit(0 if urllib.request.urlopen(url, timeout=4).status == 200 else 1)"

CMD ["python", "main.py", "--mode", "api"]
