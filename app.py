from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
import time
from datetime import datetime
from contextlib import asynccontextmanager

from core.models import *
from core.context import context_manager, Message

from services.functions import function_manager

from client import vllm_client, VLLM_API_URL, VLLM_MODEL

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)



async def generate_with_functions(user_id: str, role: str, system_prompt: str, request: GenerateRequest):
    start_time = time.time()
    
    context = await context_manager.get_or_create_context(user_id, role)
    
    await context_manager.add_or_update_system_message(user_id, system_prompt + await function_manager.get_system_prompt_addition(), role)
    
    function_manager.set_current_image(request.image)
    
    if request.image:
        logger.info(f"🖼️ Изображение прикреплено и доступно для функций")
    
    logger.info(f"🔄 Генерация для {user_id} (роль: {role})")
    
    messages = context.get_conversation_history()
    
    final_content = ""
    all_functions_used = []
    all_function_results = {}
    max_iterations = request.max_iterations 
    iteration_count = 0
    
    current_messages = messages.copy()
    
    while iteration_count < max_iterations:
        iteration_count += 1
        logger.info(f"🔄 Итерация {iteration_count}/{max_iterations}")
        
        result = await vllm_client.generate(
            messages=current_messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty
        )
        
        if not result["success"]:
            raise HTTPException(status_code=503, detail=result["error"])
        
        response_data = result["result"]
        choice = response_data["choices"][0]
        message = choice["message"]
        raw_content = message.get("content", "")
        reasoning_content = message.get("reasoning_content", "")

        is_need_functions = await function_manager.is_need_functions(raw_content)
        
        if is_need_functions:
            logger.info(f"🔧 Найдены вызовы функций в итерации {iteration_count}")
            
            function_manager.set_current_user_id(user_id)
            
            processed_text, functions_used, function_results = await function_manager.parse_and_execute(raw_content)
            
            all_functions_used.extend(functions_used)
            all_function_results.update(function_results)
            
            content_without_calls = await function_manager.remove_function_calls(raw_content)
            
            current_messages.append({
                "role": "assistant", 
                "content": content_without_calls
            })
            
            current_messages.append({
                "role": "system",
                "content": f"Результаты выполнения функций:\n{processed_text}\n\nПродолжи ответ пользователю, используя эти данные."
            })
            
        else:
            final_content = raw_content
            logger.info(f"✅ Генерация завершена после {iteration_count} итераций")
            break
    
    if iteration_count >= max_iterations:
        logger.warning(f"⚠️ Достигнуто максимальное количество итераций ({max_iterations})")
        
        if not final_content:
            final_content = raw_content
    
    processing_time = time.time() - start_time
    clean_final_content = await function_manager.remove_function_calls(final_content)
    
    function_manager.set_current_image(None)
    function_manager.set_current_user_id(None)
    
    return {
        "content": clean_final_content,
        "reasoning_content": reasoning_content,
        "functions_used": all_functions_used,
        "function_results": all_function_results,
        "iterations": iteration_count,
        "processing_time": processing_time,
        "model": result.get("model", VLLM_MODEL),
        "usage": result.get("usage", {})
    }

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Запуск API...")
    
    if await vllm_client.health_check():
        logger.info(f"✅ API доступен: {VLLM_API_URL}")
    else:
        logger.warning(f"⚠️ API недоступен: {VLLM_API_URL}")
    
    yield
    
    logger.info("🛑 Остановка API...")

app = FastAPI(
    title="LLM API",
    description="т бааанк BEST",
    version="0.0.1",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", response_model=HealthResponse)
async def health(): 
    vllm_status = await vllm_client.health_check()
    
    return HealthResponse(
        status="healthy" if vllm_status else "offline",
        model_status="connected" if vllm_status else "disconnected",
        vllm_url=VLLM_API_URL,
        active_contexts=len(context_manager.contexts)
    )

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    try:
        logger.info(f"📝 Запрос от {request.user_id} (роль: {request.role})")

        await context_manager.add_user_message(request.user_id, request.message, request.role)
        
        result = await generate_with_functions(request.user_id, request.role, request.system_prompt, request)
        
        text_result = result["content"]
        thinking_result = result["reasoning_content"]

        await context_manager.add_assistant_message(
            request.user_id,
            result["content"],
            request.role,
            result["functions_used"],
            result["function_results"]
        )
        
        logger.info(f"✅ Генерация завершена за {result['processing_time']:.2f}с")
        
        context = await context_manager.get_or_create_context(request.user_id, request.role)
        
        images = []
        for func_name, func_result in result["function_results"].items():
            if func_name in ['text_to_image', 'image-text-to-image'] and isinstance(func_result, dict):
                image_data = func_result.get('image')
                if image_data:
                    images.append(image_data)
        
        return GenerateResponse(
            message=text_result,
            thinking=thinking_result,
            user_id=request.user_id,
            role=request.role,
            function_calls=result["functions_used"],
            function_results=result["function_results"],
            images=images,
            generation_time=result["processing_time"],
            message_count=len(context.messages),
            model_used=result["model"],
            tokens_generated=result["usage"].get("completion_tokens", 0),
            prompt_tokens=result["usage"].get("prompt_tokens", 0)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Ошибка генерации для {request.user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка генерации: {str(e)}")

@app.post("/context/info", response_model=ContextInfoResponse)
async def get_context_info(request: ContextInfoRequest):
    context_info = await context_manager.get_context_info(request.user_id, request.role)
    
    if not context_info:
        return ContextInfoResponse(
            user_id=request.user_id,
            role="unknown",
            message_count=0,
            created_at="",
            last_activity="",
            exists=False
        )
    
    return ContextInfoResponse(
        user_id=context_info["user_id"],
        role=context_info["role"],
        message_count=context_info["message_count"],
        created_at=context_info["created_at"],
        last_activity=context_info["last_activity"],
        exists=True
    )

@app.post("/context/clear", response_model=ClearContextResponse)
async def clear_context(request: ClearContextRequest):
    try:
        await context_manager.clear_context(request.user_id, request.role)
        role_str = f" (роль: {request.role})" if request.role else " (все роли)"
        
        return ClearContextResponse(
            user_id=request.user_id,
            success=True,
            message=f"Контекст пользователя {request.user_id}{role_str} успешно очищен"
        )
    except Exception as e:
        logger.error(f"❌ Ошибка очистки контекста {request.user_id}: {e}")
        return ClearContextResponse(
            user_id=request.user_id,
            success=False,
            message=f"Ошибка очистки контекста: {str(e)}"
        )

@app.post("/context/history", response_model=ConversationHistoryResponse)
async def get_conversation_history(request: ConversationHistoryRequest):
    if request.role:
        context_key = context_manager._get_context_key(request.user_id, request.role)
        context = context_manager.contexts.get(context_key)
        
        if not context:
            context = await context_manager._load_context(request.user_id, request.role)
    else:
        context = None
        for key, ctx in context_manager.contexts.items():
            if ctx.user_id == request.user_id:
                context = ctx
                break
    
    if not context:
        raise HTTPException(status_code=404, detail=f"Контекст пользователя {request.user_id} не найден")
    
    messages = context.messages[-request.limit:] if len(context.messages) > request.limit else context.messages
    
    conversation_messages = [
        ConversationMessage(
            role=msg.role,
            content=msg.content,
            timestamp=datetime.fromtimestamp(msg.timestamp).isoformat(),
            function_calls=msg.function_calls or []
        )
        for msg in messages
    ]
    
    return ConversationHistoryResponse(
        user_id=context.user_id,
        role=context.role,
        messages=conversation_messages,
        total_messages=len(context.messages),
        context_created_at=datetime.fromtimestamp(context.created_at).isoformat()
    )

@app.post("/create_avatar", response_model=CreateAvatarResponse)
async def create_avatar(request: CreateAvatarRequest):
    try:
        logger.info(f"🎨 Запрос на создание аватара: {request.json_data}")
        
        result = await function_manager.create_avatar(request.json_data)
        
        return CreateAvatarResponse(
            success=result["success"],
            prompt=result["prompt"],
            image=result["image"],
            error=result["error"]
        )
        
    except Exception as e:
        logger.error(f"❌ Ошибка создания аватара: {e}")
        return CreateAvatarResponse(
            success=False,
            prompt="",
            image=None,
            error=f"Ошибка создания аватара: {str(e)}"
        )

@app.post("/create_characters", response_model=CreateCharactersResponse)
async def create_characters(request: CreateCharactersRequest):
    try:
        logger.info(f"👤 Запрос на создание персонажа: {request.json_data}")
        
        result = await function_manager.create_characters(request.json_data)
        
        return CreateCharactersResponse(
            success=result["success"],
            system_prompt=result["system_prompt"],
            init_message=result["init_message"],
            subtitle=result["subtitle"],
            error=result["error"]
        )
        
    except Exception as e:
        logger.error(f"❌ Ошибка создания персонажа: {e}")
        return CreateCharactersResponse(
            success=False,
            system_prompt="",
            init_message="",
            subtitle="",
            error=f"Ошибка создания персонажа: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081)