FROM python:3.10

# Install system dependencies (for cv2)
RUN apt-get update && \
    apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*


# Define build arguments with default placeholders
ARG user=defaultuser
ARG group=defaultgroup
ARG uid=1000
ARG gid=1000

# Set user and group
RUN groupadd -g ${gid} ${group}
RUN useradd -u ${uid} -g ${group} -s /bin/false -m ${user}

WORKDIR /code

# change directory permission and switch user
RUN chown -R ${user}:${group} /code
USER ${user}

COPY --chown=${user}:${group} ./requirements.txt /code/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade -r ./requirements.txt

COPY --chown=${user}:${group} ./.env.prod /code/.env
COPY --chown=${user}:${group} ./app /code/app

EXPOSE 8000

# Starting API
CMD ["uvicorn", "app.main:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "8000"]