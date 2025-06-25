FROM python:3.8-slim

RUN apt update -y && apt install awscli -y
WORKDIR /app

COPY . /app
RUN pip install --no-cache-dir --default-timeout=3000 -r requirements.txt

RUN chmod +x start.sh

EXPOSE 8080

# Utilise le script comme commande de démarrage
CMD ["./start.sh"]