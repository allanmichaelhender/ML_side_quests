FROM python:3.11-slim

WORKDIR /app

# Install PyTorch (CPU-only)
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8507

# Default: launch the Streamlit dashboard
CMD ["streamlit", "run", "app.py", "--server.port=8507", "--server.address=0.0.0.0"]
