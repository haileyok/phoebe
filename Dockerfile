FROM python:3.12-slim AS base

# Install Deno (needed for the sandboxed tool executor)
RUN apt-get update && apt-get install -y --no-install-recommends curl unzip \
    && curl -fsSL https://deno.land/install.sh | sh \
    && apt-get purge -y curl unzip && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

ENV DENO_DIR=/deno
ENV PATH="/root/.deno/bin:${PATH}"

WORKDIR /app

# Install Python deps first (cache layer)
COPY pyproject.toml .
RUN pip install --no-cache-dir .

# Copy application code
COPY . .

# Pre-cache Deno dependencies
RUN deno cache src/tools/deno/runtime.ts || true

EXPOSE 8080

# Default: run the arena server
CMD ["python", "main.py", "arena", "--no-dev-mode"]
