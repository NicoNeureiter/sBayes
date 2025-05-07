# Use an official lightweight Python image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libproj-dev \
    proj-data \
    proj-bin \
    libgeos-dev \
    gdal-bin \
    libgdal-dev \
    python3-tk \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python tooling early to benefit from caching
RUN pip install --upgrade pip setuptools wheel cython

# Copy project files
COPY .. /app

# Install Python dependencies + your package
RUN pip install .

# Install sBlot (sBayes' plotting library)
RUN pip install git+https://github.com/meavia-jing/sBlot

# Optional: install test tools
RUN pip install flake8 pytest

# Set default command (can be changed when running)
CMD ["sbayes"]
