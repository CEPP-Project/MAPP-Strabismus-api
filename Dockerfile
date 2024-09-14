FROM python:3.10

# Install system dependencies (for cv2)
RUN apt-get update && \
    apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code
COPY ./requirements.txt /code/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade -r ./requirements.txt

COPY ./.env /code/.env
COPY ./app /code/app

# Starting API
CMD ["uvicorn", "app.main:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "8000"]