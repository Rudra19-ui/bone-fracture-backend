# Use official Python runtime as base
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV DJANGO_SETTINGS_MODULE server.settings

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . /app/

# Make start script executable
RUN chmod +x /app/start.sh

# Collect static files
RUN python manage.py collectstatic --noinput

# Expose port
EXPOSE 10000

# Create persistent data directory
RUN mkdir -p /data

# Start using script
CMD ["/app/start.sh"]
