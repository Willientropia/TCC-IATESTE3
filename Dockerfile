FROM python:3.10-slim

WORKDIR /app

# Instalar dependências
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar a pasta backend
COPY backend/rl /app/backend/rl

# Definir o fuso horário (opcional, ajustado para o BR)
ENV TZ="America/Sao_Paulo"
ENV PYTHONUNBUFFERED=1

# Executar a ponte
CMD ["python", "backend/rl/ai_bridge.py"]
