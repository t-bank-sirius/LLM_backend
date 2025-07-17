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
        self.current_user_id: Optional[str] = None
        self.validator = FunctionValidator()
        self._register_builtin_functions()
        
    def set_current_image(self, image: Optional[str]):
        self.current_image = image
        
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
    
    def is_image_attached(self) -> bool:
        if not self.current_image:
            return False
        return True
    
    def get_current_date(self) -> str:
        return datetime.now().strftime("%Y-%m-%d")
    
    def get_datetime(self) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def get_weekday(self) -> str:
        weekdays = ["понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"]
        return weekdays[datetime.now().weekday()]
    
    async def describe_image(self, prompt: str = "") -> str:
        if not self.current_image:
            return "❌ Изображение не предоставлено в запросе. Для анализа изображения пользователь должен прикрепить изображение к сообщению."
        try:
            result = await vlm_client.describe_image(self.current_image, prompt)
            return result
        except Exception as e:
            return f"❌ Ошибка анализа изображения: {str(e)}"
    
    async def add_face_to_db(self, text: str) -> str:
        if not self.current_image:
            return "❌ Изображение не предоставлено в запросе. Для сохранения лица пользователь должен прикрепить изображение к сообщению."
        try:
            result = await vlm_client.save_face_to_db(self.current_image, text)
            return result
        except Exception as e:
            return f"❌ Ошибка сохранения лица: {str(e)}"

    async def text_to_image(self, prompt: str, style_key: str) -> Dict[str, Any]:
        try:
            result = await gen_api.prompt_to_image(prompt, style_key)
            if isinstance(result, dict) and 'image' in result:
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
        if not self.current_image:
            return {
                "context_message": "❌ Для преобразования изображения пользователь должен прикрепить исходное изображение к сообщению.",
                "image": None,
                "success": False
            }
        try:
            result = await gen_api.image_to_image(self.current_image, prompt, style_key)
            if isinstance(result, dict) and 'image' in result:
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
            image_result = await self.text_to_image(generated_prompt, "3D Cartoon")
            
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
                    'FUNCTION_CALL:text_to_image("котенок играет с мячиком", "3D Cartoon")',
                    'FUNCTION_CALL:text_to_image("футуристический город", "3D Futuristic Scify")'
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
                    'FUNCTION_CALL:image_text_to_image("сделать более ярким и красочным", "2D Cartoon")',
                    'FUNCTION_CALL:image_text_to_image("добавить снег и зиму", "3D Hyperrealistic")'
                ]
            )
        )
        
        self.register_function(
            self.is_image_attached,
            FunctionSchema(
                name="is_image_attached",
                description="Проверить, прикреплено ли изображение к сообщению. Возращает True или False",
                parameters={"required": []},
                examples=["FUNCTION_CALL:is_image_attached()"]
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
        
        
        logger.info(f"✅ Зарегистрировано {len(self.functions)} функций")

    async def get_system_prompt_addition(self) -> str:
        return """

🔧 ДОСТУПНЫЕ ФУНКЦИИ:

Для работы с временем:
- get_current_time() - получить текущее время (ЧЧ:ММ:СС)
- get_current_date() - получить текущую дату (ГГГГ-ММ-ДД)  
- get_datetime() - получить дату и время (ГГГГ-ММ-ДД ЧЧ:ММ:СС)
- get_weekday() - получить день недели (на русском)

Для работы с изображениями:
- describe_image(prompt: str = None) - описать прикрепленное изображение (если не передать prompt, то будет описано содержимое изображения)
- add_face_to_db(text: str) - сохранить лицо с прикрепленного изображения
- text_to_image(prompt: str, style_key: str) - сгенерировать изображение по описанию
- image_text_to_image(prompt: str, style_key: str) - преобразовать прикрепленное изображение
- is_image_attached() - проверить, прикреплено ли изображение к сообщению (для тебя не видно это и так что всегда вызывай эту функцию перед использованием других функций с изображениями что бы быть уверенным что изображение есть)

Для работы с памятью:
- save_memory(content: str, context: str = None) - сохранить память в базу данных. Обязательно передавай параметр content. и передавай именно вот так save_memory(content="мое имя степан", context="имя пользователя") то есть с параметром content и context
- get_memory(query: str) - получить память из базы данных. Обязательно передавай параметр query. и передавай именно вот так get_memory(query="мое имя степан") то есть с параметром query

ДОСТУПНЫЕ СТИЛИ (style_key):
• "2D FairyTale" - 2D сказочный стиль, минималистичная иллюстрация
• "3D FairyTale" - 3D сказочная сцена с магической атмосферой
• "2D Futuristic Cyberpunk" - 2D киберпанк, дистопия будущего
• "3D Futuristic Scify" - 3D научная фантастика, футуристичные механизмы
• "2D Soft Dreamy" - 2D мягкий мечтательный стиль, пастельные тона
• "3D Soft Dreamy" - 3D мягкий стиль, пушистые текстуры
• "2D Hyperrealistic" - 2D гиперреалистичный стиль
• "3D Hyperrealistic" - 3D гиперреалистичный стиль
• "2D Game Art" - 2D игровой арт, пиксельный стиль
• "3D Game Art" - 3D игровой арт для видеоигр
• "2D Cartoon" - 2D мультяшный стиль для детей
• "3D Cartoon" - 3D мультяшный стиль, красочный
• "2D Anime" - 2D аниме стиль, большие глаза, яркие цвета
• "3D Anime" - 3D аниме стиль, стилизованные персонажи

ПРАВИЛА ИСПОЛЬЗОВАНИЯ:
1. Формат вызова: FUNCTION_CALL:имя_функции(параметры)
2. Примеры:
   - FUNCTION_CALL:get_current_time()
   - FUNCTION_CALL:text_to_image("красивый закат", "3D Hyperrealistic")
   - FUNCTION_CALL:image_text_to_image("добавить снег", "2D Cartoon")
   - FUNCTION_CALL:describe_image()
3. Всегда используй точные названия функций и правильный синтаксис
4. Для функций генерации изображений обязательно указывай style_key из списка
5. Функции с изображениями работают только если пользователь прикрепил изображение (is_image_attached() возращает True)

ВАЖНО: 
 
- Почти всегда используй функцию save_memory для сохранения памяти, если пользователь просит сохранить что-то в память или ты так считаешь что это нужно сохранить
- Всегда используй функцию get_memory для получения памяти, если ты считаешь что это нужно для ответа, то есть почти всегда для улучшения контекста
- Вызывай функции всегда когда требуется для ответа и не выдумывай и не предполагай что это не нужно
- При ошибках функций объясни пользователю причину и как исправить
- Не предлагать пользователю использовать функции настойчиво, если он не просил о них рассказать
- Не используй из контекста результаты выполнения функций, вместо их вызова"""

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
                    # Универсальный парсер именованных параметров
                    # Ищем pattern: name="value" или name='value'
                    named_params = re.findall(r'(\w+)=(["\'])([^"\']*)\2', params_str)
                    for param_name, quote, param_value in named_params:
                        parameters[param_name] = param_value
                    
                    if named_params:
                        logger.info(f"🔧 Извлечены параметры для {function_name}: {parameters}")
                    
                    # Если именованных параметров нет, используем старый метод для обратной совместимости
                    if not parameters:
                        param_matches = re.findall(r'"([^"]*)"', params_str)
                        if len(param_matches) >= 1 and function_name in ['text_to_image', 'image_text_to_image']:
                            parameters['prompt'] = param_matches[0]
                            if len(param_matches) >= 2:
                                parameters['style_key'] = param_matches[1]
                
                result = await self.execute_function_safely(function_name, parameters)
                
                functions_used.append(function_name)
                function_results[function_name] = result.result
                
                if result.success:
                    if function_name in ['text_to_image', 'image_text_to_image'] and isinstance(result.result, dict):
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

function_manager = FunctionManager() 