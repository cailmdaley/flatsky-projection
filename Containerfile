FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

# Install curl and certificates, then clean up to keep image size small
RUN apt-get update &&     apt-get install -y --no-install-recommends curl ca-certificates &&     rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"
RUN uv pip install -r requirements.txt

COPY . .
