#!/bin/bash

set -e  # Остановка при ошибках

echo "🚀 Начинаем развертывание LLM API"

# Переменные
PROJECT_DIR="$HOME/llm-api"
USER_ID=$(id -u)
GROUP_ID=$(id -g)
VLLM_MODEL=${VLLM_MODEL:-"/model"}
API_PORT=${API_PORT:-8080}
VLLM_PORT=${VLLM_PORT:-8000}

# Создание структуры директорий в домашней папке
echo "📁 Создание структуры директорий в $PROJECT_DIR..."
mkdir -p $PROJECT_DIR/{app,data,logs}
mkdir -p $PROJECT_DIR/data/{user_contexts,models}
mkdir -p $PROJECT_DIR/logs/{api,vllm}

# Установка прав доступа
echo "🔐 Настройка прав доступа..."
chmod -R 755 $PROJECT_DIR

# Копирование файлов приложения
echo "📦 Копирование файлов приложения..."
cp -r core/ services/ client.py app.py requirements.txt Dockerfile $PROJECT_DIR/app/

# Создание .env файла
echo "⚙️ Создание конфигурации..."
cat > $PROJECT_DIR/.env << EOF
# vLLM Configuration
VLLM_API_URL=http://host.docker.internal:8000
VLLM_MODEL=/model
VLLM_TIMEOUT=120

# VLM 
VLM_API_URL=http://host.docker.internal:8001

# gen
GEN_API_URL=http://host.docker.internal:8002

# API Configuration
API_HOST=0.0.0.0
API_PORT=8080

# Logging
LOG_LEVEL=INFO
EOF

# Проверка Docker
echo "🐳 Проверка Docker..."
if ! command -v docker &> /dev/null; then
    echo "❌ Docker не установлен. Обратитесь к администратору."
    exit 1
fi

# Проверка прав на Docker
if ! docker ps &> /dev/null; then
    echo "❌ Нет прав на использование Docker."
    echo "💡 Попросите администратора добавить вас в группу docker:"
    echo "   sudo usermod -aG docker $USER"
    echo "   Затем перелогиньтесь."
    exit 1
fi

# Сборка API контейнера
echo "🔨 Сборка API контейнера..."
cd $PROJECT_DIR/app
docker build -t llm-api-$USER:latest \
    --build-arg USER_UID=$USER_ID \
    --build-arg USER_GID=$GROUP_ID \
    .

# Запуск API контейнера
echo "🚀 Запуск API контейнера..."
docker run -d \
    --name api-server-$USER \
    --restart unless-stopped \
    -p $API_PORT:8080 \
    -v $PROJECT_DIR/data/user_contexts:/app/user_contexts \
    -e VLLM_API_URL=http://host.docker.internal:8000 \
    -e VLLM_MODEL=$VLLM_MODEL \
    -e VLLM_TIMEOUT=120 \
    --add-host host.docker.internal:host-gateway \
    llm-api-$USER:latest

# Ожидание запуска API
echo "⏳ Ожидание запуска API..."
sleep 10

# Проверка API
if curl -f http://localhost:$API_PORT/health &>/dev/null; then
    echo "✅ API запущен успешно"
else
    echo "❌ Ошибка запуска API"
    docker logs api-server-$USER
    exit 1
fi

# Тест системы
echo "🧪 Тестирование системы..."
TEST_RESPONSE=$(curl -s -X POST http://localhost:$API_PORT/generate \
    -H "Content-Type: application/json" \
    -d '{
        "message": "Привет!",
        "user_id": "test_user",
        "role": "assistant",
        "system_prompt": "Ты - полезный помощник."
    }' || echo "FAILED")

if [[ "$TEST_RESPONSE" == *"message"* ]]; then
    echo "✅ Система работает корректно!"
else
    echo "⚠️ Возможны проблемы с системой"
fi