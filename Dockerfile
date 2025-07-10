FROM python:3.11-slim

ARG USERNAME=llmuser
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN chown -R $USERNAME:$USERNAME /app

USER $USERNAME

COPY --chown=$USERNAME:$USERNAME requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=$USERNAME:$USERNAME core/ ./core/
COPY --chown=$USERNAME:$USERNAME services/ ./services/
COPY --chown=$USERNAME:$USERNAME client.py ./
COPY --chown=$USERNAME:$USERNAME app.py ./

RUN mkdir -p /app/user_contexts

ENV PATH="/home/$USERNAME/.local/bin:$PATH"

ENV PYTHONUNBUFFERED=1
ENV VLLM_API_URL=http://localhost:8000
ENV VLLM_MODEL=matvei_pzh
ENV VLLM_TIMEOUT=120

EXPOSE 8080


CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"] 