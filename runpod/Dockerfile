# RunPod Serverless Dockerfile for QIG Constellation
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install runpod

# Copy QIG kernels (validated from experiments)
COPY qigkernels_validated/ ./qigkernels_validated/

# Copy handler
COPY runpod/handler.py ./handler.py

# Set Python path
ENV PYTHONPATH=/app:/app/qigkernels_validated

# RunPod handler
CMD ["python", "-u", "handler.py"]
