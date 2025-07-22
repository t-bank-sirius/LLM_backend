import re
import logging
import time
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from client import gen_api, vlm_client, memory_api
logger = logging.getLogger(__name__)

class FunctionStatus(Enum):
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    VALIDATION_ERROR = "validation_error"

@dataclass
class FunctionSchema:
    name: str
    description: str
    parameters: Dict[str, Any]
    timeout: int = 30
    examples: List[str] = field(default_factory=list)

@dataclass
class FunctionResult:
    success: bool
    result: Any
    status: FunctionStatus
    error: Optional[str] = None
    function_name: str = ""
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

class FunctionValidator:
    @staticmethod
    def validate_parameters(func_name: str, parameters: Dict[str, Any], schema: FunctionSchema) -> tuple[bool, str]:
        try:
            required_params = schema.parameters.get('required', [])
            missing_params = []
            
            for param in required_params:
                if param not in parameters:
                    missing_params.append(param)
            
            if missing_params:
                error_msg = f"Отсутствуют обязательные параметры: {', '.join(missing_params)}"
                if schema.examples:
                    error_msg += f"\nПример использования: {schema.examples[0]}"
                return False, error_msg
            
            param_props = schema.parameters.get('properties', {})
            type_errors = []
            
            for param_name, param_value in parameters.items():
                if param_name in param_props:
                    expected_type = param_props[param_name].get('type')
                    if expected_type == 'string' and not isinstance(param_value, str):
                        type_errors.append(f"{param_name} должен быть строкой")
                    elif expected_type == 'number' and not isinstance(param_value, (int, float)):
                        type_errors.append(f"{param_name} должен быть числом")
            
            if type_errors:
                return False, f"Ошибки типов параметров: {'; '.join(type_errors)}"
            
            return True, ""
            
        except Exception as e:
            return False, f"Ошибка валидации: {str(e)}"

class FunctionManager:    
    def __init__(self):
        self.functions: Dict[str, Callable] = {}
        self.schemas: Dict[str, FunctionSchema] = {}
        self.current_image: Optional[str] = None
        self.current_image_id: Optional[int] = None
        self.current_user_id: Optional[str] = None
        self.validator = FunctionValidator()
        
        self.image_context: Dict[str, List[Dict]] = {} 
        self.max_image_context = 5  
        
        self._register_builtin_functions()
    
    def _add_image_to_context(self, user_id: str, image_data: str, prompt: str = "", action: str = "generated"):
        if user_id not in self.image_context:
            self.image_context[user_id] = []
        
        id = len(self.image_context[user_id]) + 1

        self.current_image_id = id
        image_entry = {
            "id": id,
            "image": image_data,
            "prompt": prompt,
            "action": action, 
            "timestamp": time.time()
        }
        
        self.image_context[user_id].append(image_entry)
        
        if len(self.image_context[user_id]) > self.max_image_context:
            self.image_context[user_id] = self.image_context[user_id][-self.max_image_context:]
        
        logger.info(f"🖼️ Добавлено изображение в контекст для {user_id}: {action} - {prompt}")
        
        # Обновляем id для всех изображений после добавления нового
        self._update_image_ids(user_id)
    
    def _update_image_ids(self, user_id: str):
        """Обновляет id изображений после изменений в контексте"""
        if user_id in self.image_context:
            for i, img in enumerate(self.image_context[user_id], 1):
                img['id'] = i
    
    def set_current_image(self, image: Optional[str]):
        self.current_image = image
        self.current_image_id = None  
        
        if self.current_user_id and image:
            self._add_image_to_context(
                self.current_user_id, 
                image, 
                "пользователь загрузил изображение", 
                "uploaded"
            )
        
    def set_current_user_id(self, user_id: Optional[str]):
        self.current_user_id = user_id
    
    def register_function(self, func: Callable, schema: FunctionSchema):
        self.functions[schema.name] = func
        self.schemas[schema.name] = schema
        logger.info(f"✅ Зарегистрирована функция: {schema.name}")
    
    async def execute_function_safely(self, func_name: str, parameters: Dict[str, Any]) -> FunctionResult:
        start_time = time.time()
        
        function_name_mapping = {}
        
        internal_func_name = function_name_mapping.get(func_name, func_name)
        
        if internal_func_name not in self.functions:
            error_msg = f"Функция '{func_name}' не найдена. Доступные функции: {', '.join(self.schemas.keys())}"
            return FunctionResult(
                success=False,
                result=None,
                status=FunctionStatus.ERROR,
                error=error_msg,
                function_name=func_name
            )
        
        schema = self.schemas[func_name]
        
        is_valid, validation_error = self.validator.validate_parameters(func_name, parameters, schema)
        if not is_valid:
            return FunctionResult(
                success=False,
                result=None,
                status=FunctionStatus.VALIDATION_ERROR,
                error=validation_error,
                function_name=func_name
            )
        
        try:
            func = self.functions[internal_func_name]
            
            if asyncio.iscoroutinefunction(func):
                if parameters:
                    result = await asyncio.wait_for(func(**parameters), timeout=schema.timeout)
                else:
                    result = await asyncio.wait_for(func(), timeout=schema.timeout)
            else:
                if parameters:
                    result = func(**parameters)
                else:
                    result = func()
            
            execution_time = time.time() - start_time
            
            return FunctionResult(
                success=True,
                result=result,
                status=FunctionStatus.SUCCESS,
                function_name=func_name,
                execution_time=execution_time,
                metadata={'parameters': parameters}
            )
            
        except asyncio.TimeoutError:
            error_msg = f"Функция '{func_name}' превысила лимит времени ({schema.timeout}s)"
            return FunctionResult(
                success=False,
                result=None,
                status=FunctionStatus.TIMEOUT,
                error=error_msg,
                function_name=func_name,
                execution_time=time.time() - start_time
            )
        except Exception as e:
            error_msg = f"Ошибка выполнения функции '{func_name}': {str(e)}"
            logger.error(f"❌ {error_msg}")
            return FunctionResult(
                success=False,
                result=None,
                status=FunctionStatus.ERROR,
                error=error_msg,
                function_name=func_name,
                execution_time=time.time() - start_time
            )
    
    def get_current_time(self) -> str:
        return datetime.now().strftime("%H:%M:%S")
    
    
    def get_current_date(self) -> str:
        return datetime.now().strftime("%Y-%m-%d")
    
    def get_datetime(self) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def get_weekday(self) -> str:
        weekdays = ["понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"]
        return weekdays[datetime.now().weekday()]
    
    async def describe_image(self, prompt: str = "") -> str:
        logger.info(f"[DEBUG] describe_image: user_id={self.current_user_id}, current_image={'set' if self.current_image else 'None'}, current_image_id={self.current_image_id}")
        if not self.current_image:
            logger.error("[DEBUG] describe_image: current_image is None!")
            return "❌ Изображение не выбрано. Сначала используй select_image_by_id(id=...) для выбора нужного изображения."
        
        try:
            result = await vlm_client.describe_image(self.current_image, prompt)

            if isinstance(result, dict):
                if self.current_user_id and result.get('image'):
                    self._add_image_to_context(
                        self.current_user_id, 
                        result['image'], 
                        prompt, 
                        "described"
                    )
            return result
        except Exception as e:
            logger.error(f"[DEBUG] describe_image: exception: {e}")
            return f"❌ Ошибка анализа изображения: {str(e)}"
    
    async def add_face_to_db(self, text: str) -> str:
        if not self.current_image:
            return "❌ Изображение не предоставлено в запросе. Для сохранения лица пользователь должен прикрепить изображение к сообщению."
        try:
            result = await vlm_client.save_face_to_db(self.current_image, text)

            if self.current_user_id and result['image']:
                self._add_image_to_context(
                    self.current_user_id, 
                    result['image'], 
                    text, 
                    "saved"
                )
            return result
        except Exception as e:
            return f"❌ Ошибка сохранения лица: {str(e)}"

    async def text_to_image(self, prompt: str, style_key: str) -> Dict[str, Any]:
        try:
            result = await gen_api.prompt_to_image(prompt, style_key)
            if isinstance(result, dict) and 'image' in result:
                if self.current_user_id and result['image']:
                    self._add_image_to_context(
                        self.current_user_id, 
                        result['image'], 
                        prompt, 
                        "generated"
                    )
                
                return {
                    "context_message": f"🖼️ Изображение успешно сгенерировано по запросу: '{prompt}' в стиле '{style_key}'",
                    "image": result['image'],
                    "success": True
                }
            else:
                return {
                    "context_message": f"❌ Не удалось сгенерировать изображение по запросу: '{prompt}'",
                    "image": None,
                    "success": False
                }
        except Exception as e:
            return {
                "context_message": f"❌ Ошибка генерации изображения: {str(e)}",
                "image": None,
                "success": False
            }
    
    async def image_text_to_image(self, prompt: str, style_key: str) -> Dict[str, Any]:
        logger.info(f"[DEBUG] image_text_to_image: user_id={self.current_user_id}, current_image={'set' if self.current_image else 'None'}, current_image_id={self.current_image_id}")
        if not self.current_image:
            return {
                "context_message": "❌ Изображение не выбрано. Сначала используйте list_images_in_context() для просмотра доступных изображений, затем select_image_by_id(id=...) для выбора нужного изображения.",
                "image": None,
                "success": False
            }
        try:
            result = await gen_api.image_to_image(self.current_image, prompt, style_key)
            if isinstance(result, dict) and 'image' in result:
                if self.current_user_id and result['image']:
                    self._add_image_to_context(
                        self.current_user_id, 
                        result['image'], 
                        prompt, 
                        "modified"
                    )
                
                return {
                    "context_message": f"🖼️ Изображение успешно преобразовано по запросу: '{prompt}' в стиле '{style_key}'",
                    "image": result['image'],
                    "success": True
                }
            else:
                return {
                    "context_message": f"❌ Не удалось преобразовать изображение по запросу: '{prompt}'",
                    "image": None,
                    "success": False
                }
        except Exception as e:
            return {
                "context_message": f"❌ Ошибка преобразования изображения: {str(e)}",
                "image": None,
                "success": False
            }

    async def sketch_to_image(self, prompt: str) -> Dict[str, Any]:
        logger.info(f"[DEBUG] sketch_to_image: user_id={self.current_user_id}, current_image={'set' if self.current_image else 'None'}, current_image_id={self.current_image_id}")
        if not self.current_image:
            return {
                "context_message": "❌ Изображение не выбрано. Сначала используйте list_images_in_context() для просмотра доступных изображений, затем select_image_by_id(id=...) для выбора нужного изображения.",
                "image": None,
                "success": False
            }
        try:
            result = await gen_api.sketch_to_image(self.current_image, prompt)
            if isinstance(result, dict) and 'image' in result:
                if self.current_user_id and result['image']:
                    self._add_image_to_context(
                        self.current_user_id, 
                        result['image'], 
                        prompt, 
                        "sketched"
                    )
                
                return {
                    "context_message": f"🎨 Изображение успешно создано на основе скетча по запросу: '{prompt}'",
                    "image": result['image'],
                    "success": True
                }
            else:
                return {
                    "context_message": f"❌ Не удалось создать изображение на основе скетча по запросу: '{prompt}'",
                    "image": None,
                    "success": False
                }
        except Exception as e:
            return {
                "context_message": f"❌ Ошибка создания изображения на основе скетча: {str(e)}",
                "image": None,
                "success": False
            }

    async def save_memory(self, content: str, context: str = None) -> str:
        try:
            if not self.current_user_id:
                return "❌ Ошибка: не указан пользователь для сохранения памяти"
            result = await memory_api.save_memory(self.current_user_id, content, context)
            return result
        except Exception as e:
            return f"❌ Ошибка сохранения памяти: {str(e)}"
    
    async def get_memory(self, query: str) -> str:
        try:
            if not self.current_user_id:
                return "❌ Ошибка: не указан пользователь для получения памяти"
            result = await memory_api.get_memory(self.current_user_id, query)
            return result
        except Exception as e:
            return f"❌ Ошибка получения памяти: {str(e)}"

    async def create_avatar(self, json_data: str) -> Dict[str, Any]:
        try:
            avatar_prompt_system = """Ты - эксперт по созданию промптов для генерации изображений аватаров. 
            Твоя задача - преобразовать JSON данные о персонаже в детальный промпт на английском языке для генерации изображения.
            
            Создай промпт который включает:
            - Внешность персонажа (лицо, волосы, цвет глаз, телосложение)
            - Одежду и аксессуары
            - Позу и выражение лица
            - Стиль изображения (портрет, полный рост, и т.д.)
            - Качество изображения (высокое разрешение, детали)
            
            Отвечай ТОЛЬКО промптом на английском языке, без дополнительных объяснений."""
            
            messages = [
                {"role": "system", "content": avatar_prompt_system},
                {"role": "user", "content": f"Создай промпт для изображения аватара на основе этих данных: {json_data}"}
            ]
            
            from client import vllm_client
            result = await vllm_client.generate(
                messages=messages,
                max_tokens=3000,
                temperature=0.7
            )
            
            if not result["success"]:
                return {
                    "success": False,
                    "prompt": "",
                    "image": None,
                    "error": f"Ошибка генерации промпта: {result.get('error', 'Неизвестная ошибка')}"
                }
            
            generated_prompt = result["result"]["choices"][0]["message"]["content"].strip()
            image_result = await gen_api.create_avatar(generated_prompt)
            
            if image_result.get("success"):
                return {
                    "success": True,    
                    "prompt": generated_prompt,
                    "image": image_result.get("image"),
                    "error": None
                }
            else:
                return {
                    "success": False,
                    "prompt": generated_prompt,
                    "image": None,
                    "error": image_result.get("context_message", "Ошибка генерации изображения")
                }
                
        except Exception as e:
            return {
                "success": False,
                "prompt": "",
                "image": None,
                "error": f"❌ Ошибка создания аватара: {str(e)}"
            }

    async def create_characters(self, json_data: str) -> Dict[str, Any]:
        try:
            character_system = """Сгенерируй системный промпт для чат-бота, который будет играть роль персонажа и общаться с ребёнком в возрасте 7–14 лет.  
Системный промпт должен полностью описывать личность персонажа, стиль общения и правила поведения.  

Используй следующие входные данные из JSON:  
- Имя персонажа: name  
- Пол: sex  
- Аватар: shape.description  
- Интересы: interests  
- Способности и черты характера: abilities  
- Любимые места: places  
- Пожелания ребёнка: additionalDetails  

**Структура и требования к готовому системному промпту:**  

1. **Описание персонажа**  
   - Укажи имя, пол, внешний вид (аватар).  
   - Опиши личность: добрый, весёлый, любознательный (с добавлением характеристик из данных).  

2. **Интересы и любимые занятия**  
   - Включи все ключевые интересы и хобби персонажа.  

3. **Уникальные особенности**  
   - Добавь 2–3 милые или смешные черты (например: любит рифмовать слова, вставляет забавные звуки, коллекционирует смешные факты).  

4. **Любимые места**  
   - Укажи 2–3 места из списка и коротко объясни, почему персонаж их любит (1–2 слова: «уютно», «весело»).  

5. **Интеграция пожеланий ребёнка**  
   - Добавь их в характер, стиль общения или хобби персонажа (например, если ребёнок любит динозавров, персонаж иногда рассказывает истории про динозавров).  

6. **Стиль общения**  
   - Определи тон как: доброжелательный, весёлый, понятный детям 7–14 лет.  
   - Сообщения должны быть короткими (1–3 предложения), с положительными эмоциями и дружелюбной интонацией.  

7. **Правила поведения персонажа**  
   - Всегда говорит от первого лица.  
   - Поддерживает разговор, задаёт вопросы по интересам ребёнка.  
   - Избегает сложных тем (политика, деньги, насилие). Если такие темы возникают, перенаправляет к родителям («Это важно, лучше спроси у мамы или папы»).  
   - Никогда не выходит из роли.  

8. **Цель**  
   Заверши системный промпт инструкцией:  
   «Твоя задача — быть другом ребёнку и сделать разговор увлекательным и добрым».  

Выведи готовый системный промпт в формате связного текста (без маркированных списков, без упоминания шаблонных инструкций).

После системного промпта создай начальное сообщение от лица персонажа - дружелюбное приветствие, которое представляет персонажа и показывает его характер (1-2 предложения).

Также создай короткий subtitle (подзаголовок) для персонажа - одну фразу, которая описывает его ключевые черты характера (например: "Веселый мечтатель", "Добрый исследователь").

Отвечай строго в формате:
SYSTEM_PROMPT: [текст системного промпта]
INIT_MESSAGE: [текст начального сообщения]
SUBTITLE: [короткий подзаголовок персонажа]"""
            
            messages = [
                {"role": "system", "content": character_system},
                {"role": "user", "content": f"Создай системный промпт, начальное сообщение и subtitle для персонажа на основе данных: {json_data}"}
            ]
            
            from client import vllm_client
            result = await vllm_client.generate(
                messages=messages,
                max_tokens=2048,
                temperature=0.8
            )
            
            if not result["success"]:
                return {
                    "success": False,
                    "system_prompt": "",
                    "init_message": "",
                    "subtitle": "",
                    "error": f"Ошибка генерации персонажа: {result.get('error', 'Неизвестная ошибка')}"
                }
            
            generated_text = result["result"]["choices"][0]["message"]["content"].strip()
            
            import re
            system_match = re.search(r'SYSTEM_PROMPT:\s*(.*?)(?=INIT_MESSAGE:|$)', generated_text, re.DOTALL)
            init_match = re.search(r'INIT_MESSAGE:\s*(.*?)(?=SUBTITLE:|$)', generated_text, re.DOTALL)
            subtitle_match = re.search(r'SUBTITLE:\s*(.*)', generated_text, re.DOTALL)
            
            if system_match and init_match and subtitle_match:
                system_prompt = system_match.group(1).strip()
                init_message = init_match.group(1).strip()
                subtitle = subtitle_match.group(1).strip()
                
                return {
                    "success": True,
                    "system_prompt": system_prompt,
                    "init_message": init_message,
                    "subtitle": subtitle,
                    "error": None
                }
            else:
                return {
                    "success": False,
                    "system_prompt": "",
                    "init_message": "",
                    "subtitle": "",
                    "error": "❌ Не удалось распарсить ответ LLM"
                }
                
        except Exception as e:
            return {
                "success": False,
                "system_prompt": "",
                "init_message": "",
                "subtitle": "",
                "error": f"❌ Ошибка создания персонажа: {str(e)}"
            }
    
    def _register_builtin_functions(self):
        self.register_function(
            self.get_current_time,
            FunctionSchema(
                name="get_current_time",
                description="Получить текущее время в формате ЧЧ:ММ:СС",
                parameters={"required": []},
                examples=["FUNCTION_CALL:get_current_time()"]
            )
        )
        
        self.register_function(
            self.get_current_date,
            FunctionSchema(
                name="get_current_date",
                description="Получить текущую дату в формате ГГГГ-ММ-ДД",
                parameters={"required": []},
                examples=["FUNCTION_CALL:get_current_date()"]
            )
        )
        
        self.register_function(
            self.get_datetime,
            FunctionSchema(
                name="get_datetime",
                description="Получить текущую дату и время в формате ГГГГ-ММ-ДД ЧЧ:ММ:СС",
                parameters={"required": []},
                examples=["FUNCTION_CALL:get_datetime()"]
            )
        )
        
        self.register_function(
            self.get_weekday,
            FunctionSchema(
                name="get_weekday",
                description="Получить текущий день недели на русском языке",
                parameters={"required": []},
                examples=["FUNCTION_CALL:get_weekday()"]
            )
        )
        
        self.register_function(
            self.describe_image,
            FunctionSchema(
                name="describe_image",
                description="Описать содержимое прикрепленного изображения. Работает только если пользователь прикрепил изображение к сообщению. Можно передать дополнительное описание к изображению для улучшения понимания изображения. Возращает описание на английском языке и prompt также на английском языке.",
                parameters={"required": []},
                timeout=30,
                examples=["FUNCTION_CALL:describe_image()", "FUNCTION_CALL:describe_image(prompt='описание изображения')"]
            )
        )
        
        self.register_function(
            self.add_face_to_db,
            FunctionSchema(
                name="add_face_to_db",
                description="Сохранить лицо с прикрепленного изображения в базу данных. Работает только если пользователь прикрепил изображение лица.",
                parameters={"required": ["text"]},
                timeout=15,
                examples=["FUNCTION_CALL:add_face_to_db(text='это кот Степана')"]
            )
        )
        
        self.register_function(
            self.text_to_image,
            FunctionSchema(
                name="text_to_image",
                description="Сгенерировать изображение по текстовому описанию в указанном стиле",
                parameters={
                    "required": ["prompt", "style_key"],
                    "properties": {
                        "prompt": {"type": "string", "description": "Описание изображения для генерации"},
                        "style_key": {"type": "string", "description": "Стиль изображения из списка доступных стилей"}
                    }
                },
                timeout=60,
                examples=[
                    'FUNCTION_CALL:text_to_image("kitten playing with ball", "3D Cartoon")',
                    'FUNCTION_CALL:text_to_image("futuristic city", "3D Futuristic Scify")'
                ]
            )
        )
        
        self.register_function(
            self.image_text_to_image,
            FunctionSchema(
                name="image_text_to_image",
                description="Преобразовать прикрепленное изображение согласно текстовому описанию в указанном стиле",
                parameters={
                    "required": ["prompt", "style_key"],
                    "properties": {
                        "prompt": {"type": "string", "description": "Описание желаемых изменений изображения"},
                        "style_key": {"type": "string", "description": "Стиль изображения из списка доступных стилей"}
                    }
                },
                timeout=60,
                examples=[
                    'FUNCTION_CALL:image_text_to_image("make it brighter and colorful", "2D Cartoon")',
                    'FUNCTION_CALL:image_text_to_image("add snow and winter", "3D Hyperrealistic")'
                ]
            )
        )
        
        self.register_function(
            self.sketch_to_image,
            FunctionSchema(
                name="sketch_to_image",
                description="Создать изображение на основе детского рисунка/скетча/наброска. Принимает скетч и создает полноценное изображение.",
                parameters={
                    "required": ["prompt"],
                    "properties": {
                        "prompt": {"type": "string", "description": "Описание того, что должно получиться на основе скетча"}
                    }
                },
                timeout=60,
                examples=[
                    'FUNCTION_CALL:sketch_to_image("create a bright cat in cartoon style")',
                    'FUNCTION_CALL:sketch_to_image("transform into a beautiful landscape with trees")'
                ]
            )
        )
        
        
        self.register_function(
            self.save_memory,
            FunctionSchema(
                name="save_memory",
                description="Сохранить память в базу данных",
                parameters={"required": ["content"]},
                examples=["FUNCTION_CALL:save_memory(content='котенок играет с мячиком', 'контекст')"]
            )
        )
        
        self.register_function(
            self.get_memory,
            FunctionSchema(
                name="get_memory",
                description="Получить память из базы данных",
                parameters={"required": ["query"]},
                examples=["FUNCTION_CALL:get_memory(query='котенок играет с мячиком')"]
            )
        )
        
        self.register_function(
            self.list_images_in_context,
            FunctionSchema(
                name="list_images_in_context",
                description="Показать список изображений, сохраненных в контексте пользователя.",
                parameters={"required": []},
                examples=["FUNCTION_CALL:list_images_in_context()"]
            )
        )
        
        self.register_function(
            self.select_image_by_id,
            FunctionSchema(
                name="select_image_by_id",
                description="Выбрать изображение из контекста по id (после list_images_in_context). После этого describe_image будет использовать выбранное изображение.",
                parameters={"required": ["id"]},
                examples=["FUNCTION_CALL:select_image_by_id(id=2)"]
            )
        )
        
        self.register_function(
            self.get_current_image_info,
            FunctionSchema(
                name="get_current_image_info",
                description="Получить информацию о текущем выбранном изображении в контексте пользователя.",
                parameters={"required": []},
                examples=["FUNCTION_CALL:get_current_image_info()"]
            )
        )
        
        logger.info(f"✅ Зарегистрировано {len(self.functions)} функций")

    async def get_images_info(self) -> str:
        images_info = ""
        if self.current_user_id:
            images = self.image_context.get(self.current_user_id, [])
            if images:
                last_img = images[-1]
                
                if last_img['action'] == 'uploaded':
                    images_info = "\n📷 есть загруженное изображение!\n"
                elif last_img['action'] == 'generated':
                    images_info = "\n🖼️ есть сгенерированное изображение!\n"
                elif last_img['action'] == 'modified':
                    images_info = "\n🎨 есть измененное изображение!\n"
                elif last_img['action'] == 'described':
                    images_info = "\n🔍 есть описание изображения!\n"
                elif last_img['action'] == 'saved':
                    images_info = "\n👤 есть лицо с этого изображения в базе!\n"
                else:
                    images_info = "\n📋 есть изображение в контексте!\n"
                images_info += "\nФотка уже есть и доступна для работы с функциями\n"
            else:
                images_info = "\n📋 Пока нет загруженных изображений. Прикрепи фото или нарисуй что-нибудь!\n"
        else:
            images_info = "\n📋 Пользователь не указан\n"

        current_image_info = ""
        if self.current_image and self.current_image_id:
            current_image_info = "\n🎯 Сейчас выбрано изображение\n"
        else:
            current_image_info = "\n🎯 Изображение не выбрано\n"
        current_image_info += "\nФотка уже выбрана и доступна для работы с функциями\n"
        return (images_info, current_image_info)
    
    async def get_system_prompt_addition(self, character_prompt: str) -> str:
        images_info, current_image_info = await self.get_images_info()

        system_prompt = f"""
Внимательно следуй стилю написания и поведению персонажа, но И ТАКЖЕ ВНИМАТЕЛЬНО СЛЕДУЙ ПРАВИЛАМ ВЫЗОВА ФУНКЦИЙ.
[ХАРАКТЕР_И_СТИЛЬ_ПЕРСОНАЖА]  
{character_prompt}

[ОБЩИЕ_ПРАВИЛА_ВЗАИМОДЕЙСТВИЯ]
- ВСЕ ответы пользователю — ТОЛЬКО на русском языке.
- Никогда не выдавай пользователю техническую информацию (ID изображений, внутренние статусы, названия функций, кроме как в FUNCTION_CALL).
- Общайся естественно и дружелюбно.
- Если у пользователя есть загруженное изображение или ты что-то сгенерировал/изменил, СРАЗУ (без вопросов) предлагай дальнейшие действия, соответствующие контексту. Например, если пользователь прислал фото — сразу опиши его, если спросил "запомни" — сразу добавь лицо в базу.

[АЛГОРИТМ_ВЫЗОВА_ФУНКЦИЙ]
Перед вызовом ЛЮБОЙ функции ты ДОЛЖЕН строго следовать этому алгоритму:

1.  АНАЛИЗ ЗАПРОСА:
    -   Что именно хочет пользователь?
    -   Есть ли в запросе явные или неявные триггеры функций (например, "опиши фото", "запомни", "нарисуй", "сделай из рисунка")?
    -   Достаточно ли данных для вызова функции?

2.  ПРОВЕРКА КОНТЕКСТА:
    -   Проверь память (get_memory) — возможно, нужная информация уже есть.
    -   Есть ли в предыдущих сообщениях нужные данные (например, прикрепленное изображение)?

3.  ПРИНЯТИЕ_РЕШЕНИЯ_И_ДЕЙСТВИЕ:
    -   Если данных достаточно и есть явный триггер → СРАЗУ вызывай функцию.
    -   Если данных достаточно, но запрос неявный → вызывай функцию, но при этом объясни пользователю, что ты делаешь, в дружелюбной манере.
    -   Если данных мало → задай ОЧЕНЬ КОНКРЕТНЫЙ и короткий уточняющий вопрос.

4.  ОБОСНОВАНИЕ (ВНУТРЕННЕЕ):
    -   Кратко объясни себе, почему вызываешь именно эту функцию и откуда взял параметры. Это только для твоего мышления, не для пользователя.

[ЯЗЫКОВЫЕ_ПРАВИЛА]
- LLM понимает русский язык, и ты должен общаться с пользователем на русском.
- Параметры в функциях (prompt, style_key, text, query) — ТОЛЬКО на английском языке.
- При получении запроса на русском — переводи его в английский для параметров функций.
- При получении результата от функций — переводи описание на русский для пользователя.

[ИНФОРМАЦИЯ_О_КОНТЕКСТЕ_ИЗОБРАЖЕНИЙ]
{images_info}
{current_image_info}

[ДОСТУПНЫЕ_ФУНКЦИИ_И_ПРИМЕРЫ_ИСПОЛЬЗОВАНИЯ]

ВРЕМЯ И ДАТА:
- get_current_time() - текущее время (ЧЧ:ММ:СС)
- get_current_date() - текущая дата (ГГГГ-ММ-ДД)  
- get_datetime() - дата и время (ГГГГ-ММ-ДД ЧЧ:ММ:СС)
- get_weekday() - день недели (на русском)

[РАБОТА_С_ИЗОБРАЖЕНИЯМИ]
ВАЖНО: Перед вызовом describe_image, image_text_to_image или sketch_to_image — убедись, что изображение выбрано с помощью select_image_by_id! Если изображение уже выбрано (см. раздел [ИНФОРМАЦИЯ_О_КОНТЕКСТЕ_ИЗОБРАЖЕНИЙ]), то сразу используй его.

[ОПИСАНИЕ_ИЗОБРАЖЕНИЯ]
- describe_image(prompt: str = ""): Описать содержимое прикрепленного изображения.
  Примеры: FUNCTION_CALL:describe_image(prompt="a fluffy cat")
- add_face_to_db(text: str): Сохранить лицо с прикрепленного изображения в базу данных.
  Примеры: FUNCTION_CALL:add_face_to_db(text="Grisha")
- text_to_image(prompt: str, style_key: str): Сгенерировать изображение по текстовому описанию в указанном стиле.
  Примеры: FUNCTION_CALL:text_to_image(prompt="a cute kitten playing", style_key="3D Cartoon")
- image_text_to_image(prompt: str, style_key: str): Преобразовать прикрепленное изображение согласно текстовому описанию в указанном стиле.
  Примеры: FUNCTION_CALL:image_text_to_image(prompt="make it brighter and more colorful", style_key="2D Cartoon")
- sketch_to_image(prompt: str): Создать изображение на основе детского рисунка/скетча/наброска.
  Примеры: FUNCTION_CALL:sketch_to_image(prompt="transform into a beautiful landscape")
- list_images_in_context(): Показать список сохраненных изображений.
- select_image_by_id(id: int): Выбрать изображение из контекста по ID для последующих операций.
  Примеры: FUNCTION_CALL:select_image_by_id(id=2)
- get_current_image_info(): Получить информацию о текущем выбранном изображении.

[ПАМЯТЬ]
ВАЖНО! Память работает только на русском языке. При сохранении или получении информации используй только русский язык.

- save_memory(content: str, context: str = ""): Сохранить информацию в память.
  Примеры: FUNCTION_CALL:save_memory(content="меня зовут Иван", context="имя пользователя")
- get_memory(query: str): Найти информацию в памяти по запросу.
  Примеры: FUNCTION_CALL:get_memory(query="имя пользователя")

[ИНИЦИАТИВНЫЕ_СЦЕНАРИИ_И_ПРИМЕРЫ]

Сценарий 1: Пользователь присылает фото с вопросом о человеке.
Пользователь: "а этого человека помнишь? [image]"
ТВОЁ ДЕЙСТВИЕ: FUNCTION_CALL:describe_image() # Сразу описываешь изображение.

Сценарий 2: Пользователь рассказывает о друге, затем присылает фото.
Пользователь: "у меня есть друг - гриша"
ТВОЁ ДЕЙСТВИЕ: FUNCTION_CALL:save_memory(content="у меня есть друг - гриша", context="информация о друге")
Пользователь: "а вот кстати и он [image]"
ТВОЁ ДЕЙСТВИЕ: FUNCTION_CALL:add_face_to_db(text="Grisha") # Сразу добавляешь лицо в базу, используя имя из памяти.

Сценарий 3: Пользователь просит сделать что-то с рисунком.
Пользователь: "сделай из рисунка красивую картинку"
ТВОЁ ДЕЙСТВИЕ: FUNCTION_CALL:sketch_to_image(prompt="create a beautiful picture")

Сценарий 4: Пользователь просит описать фото.
Пользователь: "опиши фото"
ТВОЁ ДЕЙСТВИЕ: FUNCTION_CALL:describe_image()

Сценарий 5: Пользователь просит изменить фото.
Пользователь: "измени фото: сделай ярче"
ТВОЁ ДЕЙСТВИЕ: FUNCTION_CALL:image_text_to_image(prompt="make it brighter", style_key="3D Cartoon") # Или другой подходящий стиль.

Сценарий 6: Пользователь просит запомнить информацию.   
Пользователь: "запомни, что меня зовут Иван"
ТВОЁ ДЕЙСТВИЕ: FUNCTION_CALL:save_memory(content="меня зовут Иван", context="имя пользователя")

Сценарий 7: Пользователь спрашивает о себе (информация из памяти).
Пользователь: "как меня зовут?"
ТВОЁ ДЕЙСТВИЕ: FUNCTION_CALL:get_memory(query="имя пользователя")

Сценарий 8: Пользователь прислал фото без конкретного запроса.
Пользователь: "[image]" (просто прикрепил фото)
ТВОЁ ДЕЙСТВИЕ: FUNCTION_CALL:describe_image() # Сразу описываешь изображение, проявляя инициативу.

[ВАЖНОЕ_ПРАВИЛО_ИНИЦИАТИВЫ]
Если пользователь прикрепил изображение, и нет явного запроса на другую функцию, всегда инициируй его описание с помощью `describe_image()`. Это ключевой момент для активного взаимодействия с изображениями.
"""
        return system_prompt

    async def parse_and_execute(self, text: str) -> tuple[str, List[str], Dict]:
        pattern = r'FUNCTION_CALL:([\w-]+)\(([^)]*)\)'
        matches = re.findall(pattern, text)

        functions_used = []
        function_results = {}
        results_output = []

        if not matches:
            return "", [], {}

        logger.info(f"🔧 Найдено {len(matches)} вызовов функций: {[match[0] for match in matches]}")

        for function_name, params_str in matches:
            try:
                parameters = {}
                if params_str.strip():
                    named_params = re.findall(r'(\w+)=(["\'])([^"\']*)\2', params_str)
                    for param_name, quote, param_value in named_params:
                        parameters[param_name] = param_value
                    

                    numeric_params = re.findall(r'(\w+)=(\d+)', params_str)
                    for param_name, param_value in numeric_params:
                        parameters[param_name] = int(param_value)
                    
                    if named_params or numeric_params:
                        logger.info(f"🔧 Извлечены параметры для {function_name}: {parameters}")
                    
                    if not parameters:
                        param_matches = re.findall(r'"([^"]*)"', params_str)
                        if len(param_matches) >= 1 and function_name in ['text_to_image', 'image_text_to_image', 'sketch_to_image']:
                            parameters['prompt'] = param_matches[0]
                            if len(param_matches) >= 2 and function_name in ['text_to_image', 'image_text_to_image']:
                                parameters['style_key'] = param_matches[1]
                
                result = await self.execute_function_safely(function_name, parameters)
                
                functions_used.append(function_name)
                function_results[function_name] = result.result
                
                if result.success:
                    if function_name in ['text_to_image', 'image_text_to_image', 'sketch_to_image'] and isinstance(result.result, dict):
                        context_message = result.result.get('context_message', str(result.result))
                        results_output.append(f"✅ {function_name}: {context_message}")
                        logger.info(f"✅ Выполнена функция {function_name}: {context_message}")
                    else:
                        results_output.append(f"✅ {function_name}: {result.result}")
                        logger.info(f"✅ Выполнена функция {function_name}: {result.result}")
                else:
                    results_output.append(f"❌ {function_name}: {result.error}")
                    logger.error(f"❌ Ошибка функции {function_name}: {result.error}")
                         
            except Exception as e:
                error_msg = f"Ошибка обработки вызова функции {function_name}: {str(e)}"
                logger.error(f"❌ {error_msg}")
                results_output.append(f"❌ {function_name}: {error_msg}")
                function_results[function_name] = error_msg

        processed_text = "\n".join(results_output)
        return processed_text, functions_used, function_results
    
    async def is_need_functions(self, text: str) -> bool:
        pattern = r'FUNCTION_CALL:([\w-]+)\('
        matches = re.findall(pattern, text)
        return len(matches) > 0
    
    async def remove_function_calls(self, text: str) -> str:
        pattern = r'FUNCTION_CALL:([\w-]+)\([^)]*\)'
        return re.sub(pattern, '', text).strip()

    async def select_image_by_id(self, id: int) -> str:
        if not self.current_user_id:
            return "❌ Ошибка: не указан пользователь"
        images = self.image_context.get(self.current_user_id, [])
        for img in images:
            if img['id'] == id:
                self.current_image = img['image']
                self.current_image_id = id 
                logger.info(f"[DEBUG] select_image_by_id: выбран id={id}, action={img['action']}, prompt={img['prompt']}")
                return f"✅ Изображение с id={id} выбрано: [{img['action']}] '{img['prompt']}'"
        logger.error(f"[DEBUG] select_image_by_id: нет изображения с id={id}")
        return f"❌ Нет изображения с id={id}"

    async def list_images_in_context(self) -> str:
        if not self.current_user_id:
            return "❌ Ошибка: не указан пользователь"
        images = self.image_context.get(self.current_user_id, [])
        if not images:
            return "❌ В контексте нет сохраненных изображений"
        result = "📋 Изображения в контексте:\n"
        for img in images:
            time_ago = int(time.time() - img['timestamp'])
            if time_ago < 60:
                time_str = f"{time_ago}с назад"
            elif time_ago < 3600:
                time_str = f"{time_ago//60}м назад"
            else:
                time_str = f"{time_ago//3600}ч назад"
            result += f"id={img['id']}: [{img['action']}] '{img['prompt']}' ({time_str})\n"
        return result

    async def get_current_image_info(self) -> str:
        if not self.current_user_id or not self.current_image:
            return "❌ Нет выбранного изображения"
        images = self.image_context.get(self.current_user_id, [])
        for img in images:
            if img['image'] == self.current_image:
                return f"Текущее изображение: id={img['id']}, [{img['action']}] '{img['prompt']}'"
        return "❌ Текущее изображение не найдено в контексте"

function_manager = FunctionManager() 