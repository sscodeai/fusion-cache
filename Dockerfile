FROM python:3.11-slim

WORKDIR /app

# Install build deps, then the package with gateway + redis extras.
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir -e ".[gateway,redis]"

# Non-root user for safety.
RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8000

ENV FUSION_HOST=0.0.0.0 \
    FUSION_PORT=8000

CMD ["fusion-cache", "serve"]
