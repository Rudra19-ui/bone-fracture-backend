#!/bin/bash

# Exit on error
set -e

echo "Running migrations..."
python manage.py migrate --noinput

echo "Starting Gunicorn..."
gunicorn --bind 0.0.0.0:10000 server.wsgi:application
