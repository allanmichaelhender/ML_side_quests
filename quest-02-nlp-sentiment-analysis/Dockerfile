FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8502

CMD ["python", "src/train.py"]
