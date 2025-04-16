# Use Python 3.9 as the base image
FROM python:3.12-slim

# Install curl and other dependencies
RUN apt-get update && apt-get install -y curl && \
    curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s /root/.local/bin/poetry /usr/local/bin/poetry && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy Poetry files
COPY pyproject.toml poetry.lock ./

# Install dependencies with Poetry
RUN poetry install --no-root

# Copy the rest of the application
COPY . .

# Expose the port the app runs on
EXPOSE 8007

ARG TOKEN_ID
ARG TOKEN_SECRET

RUN poetry run modal token set --token-id $TOKEN_ID --token-secret $TOKEN_SECRET
# RUN poetry run modal token set --token-id ak-pqzsP301YIHh3uoi4i9Z6o --token-secret as-dSphzs8yIzPj75iBzQFZWT
# Command to run the application
CMD ["poetry", "run", "modal", "run", "modal_service.py"]