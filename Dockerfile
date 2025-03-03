FROM python:3.10

# Set environment variables for Hugging Face, Transformers, and Matplotlib caches
ENV HF_HOME=/tmp/huggingface_cache \
    TRANSFORMERS_CACHE=/tmp/transformers_cache \
    MPLCONFIGDIR=/tmp/matplotlib_cache

# Pass Hugging Face token as a build argument
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

# Install necessary system dependencies
RUN apt-get update && apt-get install -y \
    libdbus-1-dev \
    libgirepository1.0-dev \
    libcairo2-dev \
    python3-dev \
    fontconfig \
    && rm -rf /var/lib/apt/lists/*

# Create writable directories for output and cache
RUN mkdir -p /app/output /tmp/huggingface_cache /tmp/transformers_cache /tmp/matplotlib_cache \
    && chmod -R 777 /app/output /tmp/huggingface_cache /tmp/transformers_cache /tmp/matplotlib_cache 

# Copy the requirements.txt into the container
COPY requirements.txt /tmp/

# Install Python dependencies
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy source code into the container
COPY . /app

# Set the working directory to the application directory
WORKDIR /app

# Ensure the preprocessed datasets are included
COPY processed_test_dataset /app/processed_test_dataset
COPY processed_dataset /app/processed_dataset
COPY tokenized_utterances /app/tokenized_utterances

# Optional: Set environment variables for better TensorFlow logging
ENV TF_CPP_MIN_LOG_LEVEL=2

# Command to run training and keep the app alive
CMD ["python", "rp_train_opt.py"]
