FROM python:3.11-slim

WORKDIR /app

# Install system deps (OpenCV needs libgl1, kagglehub needs git)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps (CPU-only torch)
COPY requirements.txt .
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Streamlit default port
EXPOSE 8501

# Default: train the model
CMD ["python", "src/train.py"]
