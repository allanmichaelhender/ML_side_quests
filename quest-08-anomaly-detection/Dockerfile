FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for NumPy, scikit-learn
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8508

CMD ["streamlit", "run", "app.py", "--server.port=8508", "--server.address=0.0.0.0"]
