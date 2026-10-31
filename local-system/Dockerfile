FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY model.py .
COPY train.py .

RUN mkdir -p /app/data /app/checkpoints

CMD ["python", "train.py"]
