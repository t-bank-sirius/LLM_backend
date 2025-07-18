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
        
        self.image_context: Dict[str, List[Dict]] = {} 
        self.max_image_context = 5  
        
        self._register_builtin_functions()
    
    def _add_image_to_context(self, user_id: str, image_data: str, prompt: str = "", action: str = "generated"):
        if user_id not in self.image_context:
            self.image_context[user_id] = []
        
        image_entry = {
            "image": image_data,
            "prompt": prompt,
            "action": action, 
            "timestamp": time.time()
        }
        
        self.image_context[user_id].append(image_entry)
        
        if len(self.image_context[user_id]) > self.max_image_context:
            self.image_context[user_id] = self.image_context[user_id][-self.max_image_context:]
        
        logger.info(f"🖼️ Добавлено изображение в контекст для {user_id}: {action} - {prompt}")
    
    def set_current_image(self, image: Optional[str]):
        self.current_image = image
        
        if image and self.current_user_id:
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
            return "❌ Изображение не выбрано. Используйте get_image_from_context() для выбора изображения из контекста или прикрепите новое изображение."
        
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
        if not self.current_image:
            return {
                "context_message": "❌ Для преобразования изображения пользователь должен прикрепить исходное изображение к сообщению.",
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
        
        self.register_function(
            self.get_image_from_context,
            FunctionSchema(
                name="get_image_from_context",
                description="Получить изображение из контекста по запросу. Можно указать 'последнее', 'первое', 'сгенерированное', 'загруженное', 'измененное' или ключевые слова из промпта.",
                parameters={"required": ["query"]},
                examples=["FUNCTION_CALL:get_image_from_context(query='последнее')", "FUNCTION_CALL:get_image_from_context(query='первое')", "FUNCTION_CALL:get_image_from_context(query='сгенерированное')", "FUNCTION_CALL:get_image_from_context(query='кот')"]
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
        
        logger.info(f"✅ Зарегистрировано {len(self.functions)} функций")

    async def get_system_prompt_addition(self) -> str:
        return """

🔧 ДОСТУПНЫЕ ФУНКЦИИ И ИНТЕЛЛЕКТУАЛЬНАЯ СИСТЕМА ВЫЗОВОВ:

ВАЖНО! Перед вызовом ЛЮБОЙ функции ты ДОЛЖЕН проанализировать ситуацию по следующему алгоритму:

═══════════════════════════════════════════════════════════════
🧠 АЛГОРИТМ ПРИНЯТИЯ РЕШЕНИЙ О ВЫЗОВЕ ФУНКЦИЙ:

1. АНАЛИЗ ЗАПРОСА:
   - Что именно хочет пользователь?
   - Есть ли в запросе явные или неявные триггеры функций?
   - Достаточно ли данных для вызова функции?

2. ПРОВЕРКА КОНТЕКСТА:
   - Проверь память (get_memory) - возможно, нужная информация уже есть
   - Есть ли в предыдущих сообщениях нужные данные?

3. ПРИНЯТИЕ РЕШЕНИЯ:
   - Если данных достаточно → вызывай функцию
   - Если данных мало → задай уточняющий вопрос
   - Если запрос неявный → уточни намерения пользователя

4. ОБОСНОВАНИЕ:
   - Кратко объясни, почему вызываешь именно эту функцию
   - Укажи, откуда взял параметры

═══════════════════════════════════════════════════════════════

📋 ФУНКЦИИ ПО КАТЕГОРИЯМ:

🕐 ВРЕМЯ И ДАТА:
- get_current_time() - текущее время (ЧЧ:ММ:СС)
- get_current_date() - текущая дата (ГГГГ-ММ-ДД)  
- get_datetime() - дата и время (ГГГГ-ММ-ДД ЧЧ:ММ:СС)
- get_weekday() - день недели (на русском)

ТРИГГЕРЫ: "время", "который час", "сколько времени", "какое сегодня число", "какой день", "дата"

🖼️ РАБОТА С ИЗОБРАЖЕНИЯМИ:
- is_image_attached() - проверить наличие изображения (ВСЕГДА вызывай первой!)
- describe_image(prompt: str = "") - описать изображение
- add_face_to_db(text: str) - сохранить лицо в базу
- text_to_image(prompt: str, style_key: str) - генерация изображения
- image_text_to_image(prompt: str, style_key: str) - изменение изображения
- get_image_from_context(query: str) - получить изображение из контекста по запросу
- list_images_in_context() - показать список сохраненных изображений

ТРИГГЕРЫ: "картинка", "изображение", "фото", "нарисуй", "покажи", "сгенерируй", "создай изображение", "что на фото"

🧠 ПАМЯТЬ:
- save_memory(content: str, context: str = "") - сохранить в память
- get_memory(query: str) - найти в памяти

ТРИГГЕРЫ: "запомни", "сохрани", "помнишь", "как меня зовут", "мои данные", "что я говорил"

═══════════════════════════════════════════════════════════════

✅ ПРИМЕРЫ ПРАВИЛЬНЫХ ДЕЙСТВИЙ:

🔹 Пользователь: "Сколько времени?"
   Мышление: Явный запрос времени → вызываю get_current_time()
   Действие: FUNCTION_CALL:get_current_time()

🔹 Пользователь: "Нарисуй кота в мультяшном стиле"
   Мышление: Есть описание (кот) + стиль (мультяшный) → text_to_image
   Действие: FUNCTION_CALL:text_to_image(prompt="кот", style_key="3D Cartoon")

🔹 Пользователь: "Что на этом фото?" (с изображением)
   Мышление: Запрос анализа изображения → сначала проверяю наличие, потом анализирую
   Действия: 
   1. FUNCTION_CALL:is_image_attached()
   2. FUNCTION_CALL:describe_image()

🔹 Пользователь: "Запомни, что меня зовут Иван"
   Мышление: Явный запрос сохранения → save_memory
   Действие: FUNCTION_CALL:save_memory(content="меня зовут Иван", context="имя пользователя")

🔹 Пользователь: "Как меня зовут?"
   Мышление: Нужна информация из памяти → get_memory
   Действие: FUNCTION_CALL:get_memory(query="имя пользователя")

🔹 Пользователь: "Опиши то изображение, что я загружал"
   Мышление: Нужно найти загруженное изображение → get_image_from_context, потом анализ
   Действия:
   1. FUNCTION_CALL:get_image_from_context(query="загруженное")
   2. FUNCTION_CALL:describe_image()

🔹 Пользователь: "Покажи все мои изображения"
   Мышление: Нужен список изображений в контексте → list_images_in_context
   Действие: FUNCTION_CALL:list_images_in_context()

🔹 Пользователь: "Опиши картинку с котом"
   Мышление: Нужно найти изображение с котом → get_image_from_context, потом анализ
   Действия:
   1. FUNCTION_CALL:get_image_from_context(query="кот")
   2. FUNCTION_CALL:describe_image()

═══════════════════════════════════════════════════════════════

❌ ПРИМЕРЫ НЕПРАВИЛЬНЫХ ДЕЙСТВИЙ И ИХ ИСПРАВЛЕНИЯ:

🔸 НЕПРАВИЛЬНО:
   Пользователь: "Привет!"
   Неправильно: FUNCTION_CALL:get_current_time() (зачем время?)
   Правильно: Просто поприветствовать, функции не нужны

🔸 НЕПРАВИЛЬНО:
   Пользователь: "Сделай картинку"
   Неправильно: FUNCTION_CALL:text_to_image(prompt="", style_key="3D Cartoon")
   Правильно: "Что нарисовать? Опишите, какое изображение вы хотите"

🔸 НЕПРАВИЛЬНО:
   Пользователь: "Расскажи про себя"
   Неправильно: FUNCTION_CALL:get_memory(query="про себя")
   Правильно: Рассказать о своих возможностях, функции не нужны

🔸 НЕПРАВИЛЬНО:
   Пользователь: "Измени это фото" (без фото)
   Неправильно: FUNCTION_CALL:image_text_to_image(...)
   Правильно: "Пожалуйста, прикрепите изображение для обработки"

═══════════════════════════════════════════════════════════════

🎯 ДОСТУПНЫЕ СТИЛИ (style_key):
• "2D FairyTale" - 2D сказочный стиль
• "3D FairyTale" - 3D сказочная сцена
• "2D Futuristic Cyberpunk" - 2D киберпанк
• "3D Futuristic Scify" - 3D научная фантастика
• "2D Soft Dreamy" - 2D мягкий мечтательный стиль
• "3D Soft Dreamy" - 3D мягкий стиль
• "2D Hyperrealistic" - 2D гиперреалистичный
• "3D Hyperrealistic" - 3D гиперреалистичный
• "2D Game Art" - 2D игровой арт
• "3D Game Art" - 3D игровой арт
• "2D Cartoon" - 2D мультяшный стиль
• "3D Cartoon" - 3D мультяшный стиль (по умолчанию)
• "2D Anime" - 2D аниме стиль
• "3D Anime" - 3D аниме стиль

═══════════════════════════════════════════════════════════════

🚨 КРИТИЧЕСКИ ВАЖНЫЕ ПРАВИЛА:

1. ВСЕГДА проверяй память перед ответом: FUNCTION_CALL:get_memory(query="релевантный запрос")

2. ДЛЯ ИЗОБРАЖЕНИЙ: сначала FUNCTION_CALL:is_image_attached(), потом другие функции

3. НЕ ВЫЗЫВАЙ функции "на всякий случай" - только если это явно нужно

4. ВСЕГДА уточняй недостающие параметры перед вызовом функции

5. СОХРАНЯЙ важную информацию: имена, предпочтения, факты о пользователе

6. Формат вызова: FUNCTION_CALL:имя_функции(параметры)

7. ОБЪЯСНЯЙ свои действия: "Проверяю память...", "Генерирую изображение..."

═══════════════════════════════════════════════════════════════

💡 ЭВРИСТИКИ ДЛЯ РАСПОЗНАВАНИЯ НЕЯВНЫХ ЗАПРОСОВ:

• "покажи", "продемонстрируй" + описание → text_to_image
• "что это", "опиши" + изображение → describe_image  
• "помнишь", "я говорил" → get_memory
• "сейчас", "теперь", "в данный момент" → время/дата
• "запиши", "не забудь" → save_memory
• вопросы о личной информации → сначала get_memory

ПОМНИ: Лучше уточнить, чем вызвать функцию неправильно!"""

    async def parse_and_execute(self, text: str, original_user_message: str = "") -> tuple[str, List[str], Dict]:
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
                    
                    if named_params:
                        logger.info(f"🔧 Извлечены параметры для {function_name}: {parameters}")
                    
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

    async def get_image_from_context(self, query: str = "последнее") -> str:
        """Получить изображение из контекста по запросу"""
        if not self.current_user_id:
            return "❌ Ошибка: не указан пользователь"
            
        if self.current_user_id not in self.image_context or not self.image_context[self.current_user_id]:
            return "❌ В контексте нет сохраненных изображений"
        
        images = self.image_context[self.current_user_id]
        
        query_lower = query.lower()
        
        # Ищем по ключевым словам
        if "последн" in query_lower or "недавн" in query_lower or query_lower == "последнее":
            selected_image = images[-1]
        elif "первое" in query_lower or "старое" in query_lower:
            selected_image = images[0]
        elif "сгенерир" in query_lower or "нарисован" in query_lower:
            # Ищем последнее сгенерированное
            for img in reversed(images):
                if img['action'] == 'generated':
                    selected_image = img
                    break
            else:
                return "❌ Нет сгенерированных изображений в контексте"
        elif "загруж" in query_lower or "прикреп" in query_lower:
            # Ищем последнее загруженное
            for img in reversed(images):
                if img['action'] == 'uploaded':
                    selected_image = img
                    break
            else:
                return "❌ Нет загруженных изображений в контексте"
        elif "измен" in query_lower or "модифицир" in query_lower:
            # Ищем последнее измененное
            for img in reversed(images):
                if img['action'] == 'modified':
                    selected_image = img
                    break
            else:
                return "❌ Нет измененных изображений в контексте"
        else:
            # Поиск по ключевым словам в промпте
            best_match = None
            for img in reversed(images):
                if any(word in img['prompt'].lower() for word in query_lower.split()):
                    best_match = img
                    break
            
            if best_match:
                selected_image = best_match
            else:
                selected_image = images[-1]  # Последнее по умолчанию
        
        # Устанавливаем найденное изображение как текущее
        self.current_image = selected_image['image']
        
        logger.info(f"🖼️ Выбрано изображение из контекста: {selected_image['action']} - {selected_image['prompt']}")
        
        return f"✅ Выбрано изображение: {selected_image['action']} - '{selected_image['prompt']}'. Теперь можно его анализировать или изменять."

    async def list_images_in_context(self) -> str:
        """Показать список изображений в контексте"""
        if not self.current_user_id:
            return "❌ Ошибка: не указан пользователь"
            
        if self.current_user_id not in self.image_context or not self.image_context[self.current_user_id]:
            return "❌ В контексте нет сохраненных изображений"
        
        images = self.image_context[self.current_user_id]
        
        result = "📋 Изображения в контексте:\n"
        for i, img in enumerate(images, 1):
            action_emoji = {
                'generated': '🎨',
                'uploaded': '📤', 
                'modified': '✏️'
            }.get(img['action'], '📷')
            
            time_ago = int(time.time() - img['timestamp'])
            if time_ago < 60:
                time_str = f"{time_ago}с назад"
            elif time_ago < 3600:
                time_str = f"{time_ago//60}м назад" 
            else:
                time_str = f"{time_ago//3600}ч назад"
            
            result += f"{i}. {action_emoji} {img['action']} - '{img['prompt']}' ({time_str})\n"
        
        return result

function_manager = FunctionManager() 