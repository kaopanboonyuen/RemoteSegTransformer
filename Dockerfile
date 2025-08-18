# --------------------------------------------------------
# Transformer-Based Semantic Segmentation Framework
# Author: Teerapong Panboonyuen (Kao)
# Description: Dockerfile for building and deploying the transformer-based segmentation framework
# --------------------------------------------------------

# Use official PyTorch image with CUDA (adjust version as needed)
FROM pytorch/pytorch:2.0.1-cuda11.8-cudnn8-runtime

# Set working directory inside container
WORKDIR /app

# Copy requirements and install
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project files into container
COPY . /app/

# Expose ports if needed (for web serving etc., optional)
# EXPOSE 8000

# Default command (you can override this when running the container)
# For example: to train: docker run <image> python main.py --config config.yaml --mode train
CMD ["python", "main.py", "--config", "config.yaml", "--mode", "train"]