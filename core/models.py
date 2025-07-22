from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List

class GenerateRequest(BaseModel):
    message: str = Field(..., description="Сообщение пользователя")
    user_id: str = Field(..., description="Уникальный идентификатор пользователя")
    role: str = Field(..., description="Роль/персонаж для разделения контекстов")
    system_prompt: str = Field(..., description="Системный промпт для модели")
    
    max_tokens: int = Field(default=2048, ge=100, le=8192, description="Максимальное количество токенов")
    temperature: float = Field(default=0.65, ge=0.0, le=2.0, description="Температура генерации")
    top_p: float = Field(default=0.95, ge=0.0, le=1.0, description="Top-p sampling")
    repetition_penalty: float = Field(default=1.1, ge=1.0, le=2.0, description="Штраф за повторения")
    
    image: Optional[str] = Field(None, description="Изображение в формате base64")
    
    max_iterations: int = Field(default=5, ge=1, le=5, description="Максимальное количество итераций для функций")
    
    @validator("image")
    def validate_image(cls, v):
        if v:
            try:
                if v.startswith("data:image"):
                    v = v.split(",", 1)[1]
                
                import base64
                base64.b64decode(v)
                return v
            except Exception:
                raise ValueError("Невалидное base64 изображение")
        return v

class GenerateResponse(BaseModel):
    message: str = Field(..., description="Ответ модели")
    thinking: str = Field(..., description="Мысли модели")
    user_id: str = Field(..., description="Идентификатор пользователя")
    role: str = Field(..., description="Роль/персонаж")
    
    function_calls: List[str] = Field(default=[], description="Выполненные функции")
    function_results: Dict[str, Any] = Field(default={}, description="Результаты функций")
    image: Optional[str] = Field(None, description="Сгенерированное изображение в base64")
    generation_time: float = Field(..., description="Время генерации в секундах")
    
    message_count: int = Field(..., description="Количество сообщений в контексте")
    
    model_used: str = Field(..., description="Использованная модель")
    tokens_generated: Optional[int] = Field(None, description="Количество сгенерированных токенов")
    prompt_tokens: Optional[int] = Field(None, description="Количество токенов в промпте")

class ContextInfoRequest(BaseModel):
    user_id: str = Field(..., description="Идентификатор пользователя")
    role: Optional[str] = Field(None, description="Роль/персонаж (если не указана - любая активная)")

class ContextInfoResponse(BaseModel):
    user_id: str = Field(..., description="Идентификатор пользователя")
    role: str = Field(..., description="Роль/персонаж")
    message_count: int = Field(..., description="Количество сообщений")
    created_at: str = Field(..., description="Время создания контекста")
    last_activity: str = Field(..., description="Время последней активности")
    exists: bool = Field(..., description="Существует ли контекст")

class ClearContextRequest(BaseModel):
    user_id: str = Field(..., description="Идентификатор пользователя")
    role: Optional[str] = Field(None, description="Роль/персонаж (если не указана - все роли)")

class ClearContextResponse(BaseModel):
    success: bool = Field(..., description="Успешность операции")
    message: str = Field(..., description="Сообщение")

class HealthResponse(BaseModel):
    status: str = Field(..., description="Статус системы")
    model_status: str = Field(..., description="Статус модели")
    vllm_url: str = Field(..., description="URL vLLM API")
    active_contexts: int = Field(..., description="Количество активных контекстов")

class ConversationHistoryRequest(BaseModel):
    user_id: str = Field(..., description="Идентификатор пользователя")
    role: Optional[str] = Field(None, description="Роль/персонаж (если не указана - любая активная)")
    limit: int = Field(default=50, ge=1, le=500, description="Ограничение количества сообщений")

class ConversationMessage(BaseModel):
    role: str = Field(..., description="Роль (user, assistant, system, tool)")
    content: str = Field(..., description="Содержимое сообщения")
    timestamp: str = Field(..., description="Время отправки")
    function_calls: List[str] = Field(default=[], description="Использованные функции")

class ConversationHistoryResponse(BaseModel):
    user_id: str = Field(..., description="Идентификатор пользователя")
    role: str = Field(..., description="Роль/персонаж")
    messages: List[ConversationMessage] = Field(..., description="Список сообщений")
    total_messages: int = Field(..., description="Общее количество сообщений")
    context_created_at: str = Field(..., description="Время создания контекста")

# Новые модели для create_avatar и create_characters
class CreateAvatarRequest(BaseModel):
    json_data: str = Field(..., description="JSON строка с данными для генерации аватара")

class CreateAvatarResponse(BaseModel):
    success: bool = Field(..., description="Успешность операции")
    prompt: str = Field(..., description="Сгенерированный промпт для изображения")
    image: Optional[str] = Field(None, description="Сгенерированное изображение в base64")
    error: Optional[str] = Field(None, description="Ошибка если есть")

class CreateCharactersRequest(BaseModel):
    json_data: str = Field(..., description="JSON строка с данными для создания персонажа")

class CreateCharactersResponse(BaseModel):
    success: bool = Field(..., description="Успешность операции")
    system_prompt: str = Field(..., description="Системный промпт для персонажа")
    init_message: str = Field(..., description="Начальное сообщение персонажа")
    subtitle: str = Field(..., description="Короткий подзаголовок персонажа")
    error: Optional[str] = Field(None, description="Ошибка если есть")