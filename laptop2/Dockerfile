
FROM python:3.11-slim

WORKDIR /app

# Install CPU version of PyTorch
RUN pip install --no-cache-dir torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY model.py .
COPY receiver.py .
COPY resume.py .

# Create storage directories
RUN mkdir -p /app/checkpoints /app/data

# Expose HTTP port
EXPOSE 5000

# Start the HTTP receiver
CMD ["python", "receiver.py"]
