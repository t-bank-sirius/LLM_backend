import re
import logging
import time
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from client import gen_api, vlm_client, memory_api, vllm_client
from check_swear import SwearingCheck
import json
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
        self.swearing_check = SwearingCheck(reg_pred=True, bins=3, stop_words=["питон"])
        
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

    async def _validate_character_data(self, json_data: str) -> Dict[str, Any]:
        validation_system_prompt = """Ты — строгий модератор контента для детей 7-14 лет. Твоя задача — тщательно проверить предоставленные JSON данные о персонаже на соответствие строгим правилам безопасности и возрастной цензуры.
Оцени следующие аспекты:
1.  **Возрастная пригодность (7-14 лет):** Содержит ли описание персонажа что-либо, что может быть неприемлемо или непонятно для этой возрастной группы? (например, сложные темы, взрослые концепции, слишком грубый юмор).
2.  **Безопасность и этика:** Нет ли в описании насилия, агрессии, дискриминации, нецензурной лексики, сексуального подтекста, пропаганды вредных привычек или любого другого контента, который может нанести вред ребёнку?
3.  **Полнота и адекватность данных:** Достаточно ли информации в JSON для создания полноценного и интересного персонажа? Все ли ключевые поля (имя, пол, описание, интересы, способности) заполнены или могут быть логически выведены?
4.  **Соответствие правилам:** Если JSON описывает персонажа, который не может быть создан или не соответствует вышеуказанным критериям (например, \"персонаж-убийца\", \"персонаж с нецензурной лексикой\"), ты должен это выявить.

Твой ответ должен быть ТОЛЬКО в формате JSON с двумя полями:
-   `is_valid` (boolean): `true`, если контент абсолютно безопасен, пригоден для детей 7-14 лет и данные достаточны; `false` в противном случае.
-   `reason` (string): Краткое и чёткое объяснение, почему контент не прошёл проверку. Если `is_valid` = `true`, это поле должно быть пустой строкой.

Если пользователь просит страрше 14 лет сгенерировать то не страшно и так можно 
Примеры ответов:
-   `{'is_valid': true, 'reason': ''}`
-   `{'is_valid': false, 'reason': 'Описание содержит элементы насилия, что недопустимо для детского контента.'}`
-   `{'is_valid': false, 'reason': 'Описание персонажа слишком расплывчато и не содержит достаточной информации для создания.'}`
-   `{'is_valid': false, 'reason': 'Возраст персонажа не соответствует целевой аудитории (7-14 лет).'}`

Проанализируй следующие данные о персонаже: {json_data}"""

        messages = [
            {"role": "system", "content": validation_system_prompt},
            {"role": "user", "content": f"Проверь эти данные для создания персонажа: {json_data}"}
        ]

        try:
            response = await vllm_client.generate(
                messages=messages,
                max_tokens=1000,
                temperature=0.4 
            )
            
            if not response["success"]:
                logger.error(f"Ошибка вызова LLM для валидации: {response.get('error', 'Неизвестная ошибка')}")
                return {"is_valid": False, "reason": "Ошибка внутренней валидации."}

            try:
                validation_result = json.loads(response["result"]["choices"][0]["message"]["content"].strip())
                if not isinstance(validation_result, dict) or "is_valid" not in validation_result or "reason" not in validation_result:
                    raise ValueError("Неверный формат ответа LLM валидатора.")
                return validation_result
            except json.JSONDecodeError as e:
                logger.error(f"Ошибка парсинга JSON ответа валидатора: {e}. Ответ: {response['result']['choices'][0]['message']['content']}")
                return {"is_valid": False, "reason": "Ошибка формата ответа валидатора."}
            except ValueError as e:
                logger.error(f"Ошибка валидации ответа LLM: {e}")
                return {"is_valid": False, "reason": "Ошибка структуры ответа валидатора."}

        except Exception as e:
            logger.error(f"Исключение при валидации данных персонажа: {e}")
            return {"is_valid": False, "reason": f"Непредвиденная ошибка при валидации: {str(e)}"}

    async def create_avatar(self, json_data: str) -> Dict[str, Any]:
        validation_check = await self._validate_character_data(json_data)
        if not validation_check["is_valid"]:
            return {
                "success": False,
                "prompt": "",
                "image": None,
                "error": f"{validation_check['reason']}"
            }
        
        try:
            avatar_prompt_system = """
Ты — эксперт по созданию промптов для генерации уникальных и детализированных аватаров. 
Твоя задача — на основе JSON-описания персонажа создать максимально подробный, живой и индивидуальный промпт на английском языке для генерации изображения.

[ПРАВИЛА СОЗДАНИЯ ПРОМПТА]
- Используй ВСЮ доступную информацию из JSON: age, sex, description, interests, abilities, places, additionalDetails.
- Обязательно опиши: 
  - Пол, возраст (child, teen, adult, elderly)
  - Цвет, длину и стиль волос, цвет глаз, форму лица, особенности внешности (родинки, веснушки, очки и т.д.)
  - Одежду, аксессуары, украшения, головные уборы, обувь
  - Позу, выражение лица, эмоцию (улыбка, задумчивость, удивление и т.д.)
  - Окружение/фон (где находится персонаж: парк, город, комната, природа и т.д.)
  - Атмосферу и освещение (яркое солнце, мягкий свет, вечер, закат и т.д.)
  - Стиль изображения (портрет, полный рост, аниме, реализм, пиксель-арт и т.д.)
  - Качество изображения (high resolution, ultra detailed, 8k, photorealistic и т.д.)
- Добавь уникальные детали, которые делают персонажа особенным (например: необычный аксессуар, любимая игрушка, характерная поза).
- Не используй шаблонные фразы, делай промпт живым и индивидуальным.
- Пример структуры промпта: 
  "A cheerful teenage girl with long curly red hair and green eyes, wearing a yellow sundress and a straw hat, standing in a blooming garden at sunset, holding a book, ultra detailed, high resolution, soft warm light, portrait style"
- Выдай ТОЛЬКО промпт на английском языке, без дополнительных объяснений.
"""
            
            messages = [
                {"role": "system", "content": avatar_prompt_system},
                {"role": "user", "content": f"Сгенерируй промпт для изображения аватара на основе этих данных: {json_data}"}
            ]
            
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
            
            if image_result.get("status") == "success":
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
        validation_check = await self._validate_character_data(json_data)
        if not validation_check["is_valid"]:
            return {
                "success": False,
                "system_prompt": "",
                "init_message": "",
                "subtitle": "",
                "error": f"❌ Ошибка валидации данных персонажа: {validation_check['reason']}"
            }

        try:
            character_system = """
Сгенерируй системный промпт для чат-бота, который будет играть роль уникального персонажа и общаться с ребёнком 7–14 лет.

[ТРЕБОВАНИЯ К СИСТЕМНОМУ ПРОМПТУ]
- Используй ВСЮ информацию из JSON: name, sex, shape.description, interests, abilities, places, additionalDetails.
- Подробно опиши:
  - Имя, пол, внешний вид (аватар, особенности лица, одежды, аксессуары)
  - Личность: характер, привычки, манеру общения, любимые фразы, уникальные черты (например: любит шутить, всегда здоровается особым образом)
  - Интересы, хобби, любимые занятия (с примерами)
  - 2–3 милые или смешные особенности (например: коллекционирует смешные факты, вставляет в речь забавные звуки)
  - Любимые места и почему они нравятся (1–2 слова: «уютно», «весело»)
  - Пожелания ребёнка интегрируй в характер, стиль общения или хобби
  - Стиль общения: дружелюбный, весёлый, понятный детям, короткие сообщения (1–3 предложения), положительные эмоции, индивидуальные словечки
  - Приведи 2–3 любимых фразы или выражения персонажа
  - Правила поведения: всегда говорит от первого лица, поддерживает разговор, задаёт вопросы по интересам ребёнка, избегает сложных тем (политика, деньги, насилие), если такие темы — перенаправляет к родителям
- Заверши системный промпт инструкцией: 
  "Твоя задача — быть другом ребёнку и сделать разговор увлекательным и добрым".

[ТРЕБОВАНИЯ К INIT_MESSAGE]
- Придумай приветствие, которое сразу раскрывает характер персонажа, его стиль общения и индивидуальность (1–2 предложения).

[ТРЕБОВАНИЯ К SUBTITLE]
- Придумай короткий подзаголовок, который отражает уникальные черты персонажа (например: "Весёлый изобретатель", "Добрый мечтатель").

Отвечай строго в формате:
SYSTEM_PROMPT: [текст системного промпта]
INIT_MESSAGE: [текст начального сообщения]
SUBTITLE: [короткий подзаголовок персонажа]
"""
                        
            messages = [
                {"role": "system", "content": character_system},
                {"role": "user", "content": f"Создай системный промпт, начальное сообщение и subtitle для персонажа на основе данных: {json_data}"}
            ]
            
            result = await vllm_client.generate(
                messages=messages,
                max_tokens=2048,
                temperature=0.6
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
        
        self.register_function(
            self.get_face,
            FunctionSchema(
                name="get_face",
                description="Получить сохраненное лицо из базы данных по имени и установить его как текущее изображение.",
                parameters={
                    "required": ["name"],
                    "properties": {
                        "name": {"type": "string", "description": "Имя, под которым лицо было сохранено"}
                    }
                },
                examples=["FUNCTION_CALL:get_face(name='Иван')"]
            )
        )
        
        logger.info(f"✅ Зарегистрировано {len(self.functions)} функций")

    async def get_images_info(self) -> str:
        images_info = ""
        if self.current_user_id:
            images = self.image_context.get(self.current_user_id, [])
            if images and (self.current_image is None or self.current_image_id is None):
                last_img = images[-1]
                self.current_image = last_img['image']
                self.current_image_id = last_img['id']
            if images:
                images_info = "\n📋 Картинки в контексте пользователя:\n"
                for img in images:
                    action = img.get('action', 'context')
                    prompt = img.get('prompt', '')
                    desc = ""
                    if action == 'uploaded':
                        desc = f"Загружена пользователем. Описание при загрузке: '{prompt}'"
                    elif action == 'generated':
                        desc = f"Сгенерирована по промпту: '{prompt}'"
                    elif action == 'modified':
                        desc = f"Изменена по промпту: '{prompt}'"
                    elif action == 'described':
                        desc = f"Описана: '{prompt}'"
                    elif action == 'saved':
                        desc = f"Лицо сохранено с описанием: '{prompt}'"
                    else:
                        desc = f"Действие: {action}, описание: '{prompt}'"
                    images_info += f"id={img['id']}: {desc}\n"
                images_info += "\nФотки доступны для работы с функциями\n"
            else:
                images_info = "\n📋 Пока нет загруженных изображений. Прикрепи фото или нарисуй что-нибудь!\n"
        else:
            images_info = "\n📋 Пользователь не указан\n"

        current_image_info = ""
        if self.current_image and self.current_image_id:
            current_image_info = f"\n🎯 Сейчас выбрано изображение id={self.current_image_id}\n"
        else:
            current_image_info = "\n🎯 Изображение не выбрано\n"
        current_image_info += "\nФотка уже выбрана и доступна для работы с функциями\n"
        return (images_info, current_image_info)
    
    async def get_system_prompt_addition(self, character_prompt: str) -> str:
        images_info, f = await self.get_images_info()
        
        imag = await self.list_images_in_context()

        print("s", images_info, f, imag)
        system_prompt = f"""
Внимательно следуй стилю написания и поведению персонажа, но И ТАКЖЕ ВНИМАТЕЛЬНО СЛЕДУЙ ПРАВИЛАМ ВЫЗОВА ФУНКЦИЙ.
[ХАРАКТЕР_И_СТИЛЬ_ПЕРСОНАЖА]  
{character_prompt}

[ОБЩИЕ_ПРАВИЛА_ВЗАИМОДЕЙСТВИЯ]
- ВСЕ ответы пользователю — ТОЛЬКО на русском языке.
- Никогда не раскрывай технические детали (ID, названия функций, статусы), кроме случаев вызова FUNCTION_CALL.
- Общайся естественно, уверенно и дружелюбно.
- Если загружено изображение — опиши его сразу.
- Если пользователь дал команду «запомни» (или синоним) — добавь лицо в базу без лишних вопросов.
- Если ты что-то сгенерировал или изменил — сразу предложи логичное следующее действие.
- помни что лучше вызвать функцию когда не надо чем не вызывать ее вообще


[АЛГОРИТМ_ВЫЗОВА_ФУНКЦИЙ]
Перед анализом запроса на необходимость вызова функции ты ОБЯЗАН строго следовать этому алгоритму:

1.  АНАЛИЗ ЗАПРОСА:
    -   Определи, что именно хочет пользователь. 
    -   Есть ли в запросе явные или неявные триггеры на функции? (например: «опиши», «сделай», «нарисуй», «запомни», «измени», «найди»)
    -   проверяешь, Достаточно ли данных для вызова функции?

2.  ПРОВЕРКА КОНТЕКСТА:
    -   Проверь контекст. Используй контекст текущего диалога, но Если в контексте не хватает информации для полного но емкого ответа, бери информацию в долгосрочной памяти (get_memory)
    -   Есть ли в предыдущих сообщениях нужные данные (например, прикрепленное изображение)?
- Учитывай предыдущее сообщение и вложенные изображения.

3.  ПРИНЯТИЕ_РЕШЕНИЯ_И_ДЕЙСТВИЕ:
-   явный триггер + достаточно данных = вызывай функцию сразу.

- Отдавай приоритет функциям обработки изображений, если в сообщении есть картинка.

4.  ОБОСНОВАНИЕ (ВНУТРЕННЕЕ):
    -   Кратко объясни себе, почему вызываешь именно эту функцию и откуда взял параметры. Это только для твоего мышления и для того чтобы ты понимал, зачем ты это делаешь. Это не для пользователя.

Если по итогам твоего анализа запроса нет необходимости вызывать функцию, то ПЕРЕПРОВЕРЬ детально еще раз. Лучше вызывать функции, чем не вызывать.
Никогда не препдлагай пользователю спрашивать нужно ли вызвать функции, если ты уже определился что нужно, не переспрашивай его, а действуй.

[ЯЗЫКОВЫЕ_ПРАВИЛА]
- Пользователь пишет по-русски, ты тоже всегда отвечаешь по-русски.
- Параметры в функциях (prompt, style_key, text, query) — ТОЛЬКО на английском языке.
- Если пользователь даёт задание на русском — переводи его в английский внутри параметров.
- При получении результата от функций — переводи описание на русский для пользователя.
- Результат функций всегда объясняй по-русски и в том же стиле, в каком ведёшь диалог.

Внимательно проанализируй данные о контексте изображений и прикрепленных фотографий.И если есть сейчас доступная фотография, то используй ее для выполнения задания и не сообщай пользователю что есть фотография в контексте.

[ИНФОРМАЦИЯ_О_КОНТЕКСТЕ_ИЗОБРАЖЕНИЙ]
{images_info}
{imag}

[ДОСТУПНЫЕ_ФУНКЦИИ_И_ПРИМЕРЫ_ИСПОЛЬЗОВАНИЯ]

ВРЕМЯ И ДАТА:
- get_current_time() - текущее время (ЧЧ:ММ:СС)
- get_current_date() - текущая дата (ГГГГ-ММ-ДД)  
- get_datetime() - дата и время (ГГГГ-ММ-ДД ЧЧ:ММ:СС)
- get_weekday() - день недели (на русском)

[РАБОТА_С_ИЗОБРАЖЕНИЯМИ]
ВАЖНО: Перед вызовом describe_image, image_text_to_image или sketch_to_image — убедись, что изображение выбрано с помощью select_image_by_id! Если изображение уже выбрано (см. раздел [ИНФОРМАЦИЯ_О_КОНТЕКСТЕ_ИЗОБРАЖЕНИЙ]), то сразу используй его.

[ОПИСАНИЕ_ИЗОБРАЖЕНИЯ]
- describe_image(prompt: str = ""): Описать содержимое изображения. Можно передавать дополнительное описание, которое будет использоваться для описания изображения. Если не передавать prompt, то будет описано содержимое изображения. Используется для того что бы узнать что находиться на изображении и использовании для ответа пользователю.ы
  Примеры: FUNCTION_CALL:describe_image(prompt="a fluffy cat")
- add_face_to_db(text: str): Сохранить лицо с прикрепленного изображения в базу данных. Работает только если пользователь прикрепил изображение лица. Используется что бы запомнить лицо человека и использовать его для ответа пользователю. Обычно нужно сохранять лицо сообщеает что это мой друг и тд и скидывает фотографию.
  Примеры: FUNCTION_CALL:add_face_to_db(text="Grisha")
- text_to_image(prompt: str, style_key: str): Сгенерировать изображение по текстовому описанию в указанном стиле. Используется для генерации изображений по текстовому описанию. Никогда не выдывай пользователю именно какой стиль по style_key, а только описание стиля понятное для пользователя.
  Примеры: FUNCTION_CALL:text_to_image(prompt="a cute kitten playing", style_key="3D Cartoon")
- image_text_to_image(prompt: str, style_key: str): Преобразовать прикрепленное изображение согласно текстовому описанию в указанном стиле. Используется для изменения изображения согласно текстовому описанию. Никогда не выдывай пользователю именно какой стиль по style_key, а только описание стиля понятное для пользователя.
  Примеры: FUNCTION_CALL:image_text_to_image(prompt="make it brighter and more colorful", style_key="2D Cartoon")
- sketch_to_image(prompt: str): Создать изображение на основе детского рисунка/скетча/наброска. Используется для генерации изображений на основе наброска. Обычно пользователь использует эту функцию когда прикрепляет набросок и просит сделать из него красивую картинку. ключевые слова преврати набросок  
  Примеры: FUNCTION_CALL:sketch_to_image("create a bright cat in cartoon style")
- select_image_by_id(id: int): Выбрать изображение из контекста по ID для последующих операций. Если пользователь просит что то сгенерировать или изменить, то сначала вызови эту функцию и выбери изображение из контекста изображение (смотри в разделе [ИНФОРМАЦИЯ_О_КОНТЕКСТЕ_ИЗОБРАЖЕНИЙ]). Важно правильно анализировать контекст изображений и прикрепленных фотографий. Учитывай что в разделе актуальная информация о контексте изображений и прикрепленных фотографий. Никогда не сообщай пользователю что есть фотография в контексте и ты ее выбрал по id и тд.
  Примеры: FUNCTION_CALL:select_image_by_id(id=2)
- get_face(name: str): Получить сохраненное лицо из базы данных по имени и установить его как текущее изображение. Используется, когда пользователь просит сгенерировать что-то 'с собой' или 'с кем-то' по имени, и предполагается, что лицо этого человека сохранено ранее.  
  Примеры: FUNCTION_CALL:get_face(name='мое лицо'), FUNCTION_CALL:get_face(name='Иван')

[ПАМЯТЬ]
ВАЖНО! Память работает на русском языке. При сохранении или получении информации используй только русский язык.

- save_memory(content: str, context: str = ""): Сохранить информацию в память. Используется для запоминания информации о человеке или объекте. Обычно пользователь использует эту функцию когда упоминает человека или объект и просит запомнить эту информацию. И когда не просит сохранить например пишет то что мой брат отмечате день рождение 16 августа и тогда ты сохраняешь эту информацию в память. И такую подобную информацию тоже сохраняй.
  Примеры: FUNCTION_CALL:save_memory(content="меня зовут Иван", context="имя пользователя")
- get_memory(query: str): Найти информацию в памяти по запросу. Используется для получения информации из памяти. Обычно пользователь использует эту функцию когда просит вспомнить информацию о человеке или объекте. Если пользователь спрашивает какую то информацию по типу помниль ли и тд то вызывай эту функцию длч получения более детальной информации . Не пиши пользователю что ты не видишь в беседе разговора про эту информацию, а просто вызови функцию и сообщи ему результат.
  Примеры: FUNCTION_CALL:get_memory(query="имя пользователя")

Ниже описаны примеры поведения пользователя и как ты должен реагировать на запросы пользователя.

[ПОВЕДЕНЧЕСКИЕ СЦЕНАРИИ]
Сценарий 1
Пользователь присылает фото и спрашивает: "а этого помнишь?" Или что то на подобии того, где показан намек на описание фотографии.(а этого человека помнишь? и прикреплена фотография)
Твоё действие: FUNCTION_CALL:describe_image() — без лишних слов сразу распознаёшь и описываешь изображение.

Сценарий 2
Сначала пользователь упоминает человека ("у меня есть друг Гриша") или ("а вот кстати и он" и отправляет фото), потом отправляет его фото или "а это кстати Саша - мой отец" и прикрепляется фото отца.
Твоё действие:
1.  FUNCTION_CALL:save_memory(content="у меня есть друг - гриша", context="информация о друге") — фиксируешь, кого именно упомянули.
2.  FUNCTION_CALL:add_face_to_db(text="Grisha") — добавляешь лицо в базу, используя имя из памяти.
Задача: сразу вызови нужные функции, а не спрашивай пользователя что делать именно.
Сценарий 3
Пользователь просит превратить набросок в красивую финальную картинку.
Твоё действие: FUNCTION_CALL:sketch_to_image(prompt="create a beautiful picture") — обрабатываешь рисунок и создаёшь по нему полноценное изображение.

Сценарий 4
Запрос: "опиши фото" или синонимы типа "что на картинке?" или "опиши этого человека" и тд.
Твоё действие: FUNCTION_CALL:describe_image() — немедленно вызываешь функцию описания изображения.

Сценарий 5
Пользователь даёт команду вроде "сделай поярче", "измени фон", "сделай круче" или подобные намеки на изменение картинки:
Твоё действие: FUNCTION_CALL:image_text_to_image(prompt="brighter picture", style_key="3D Cartoon") — выбираешь нужный стиль, формируешь запрос, применяешь изменения.

Сценарий 6
Пользователь говорит: "запомни, что меня зовут Иван" или "дарова я артем" (или другая инфа) и похожие предложения.
Твоё действие: FUNCTION_CALL:save_memory(content="меня зовут Иван", context="имя пользователя") — сохраняешь факт в долговременную память.

Сценарий 7
Фраза пользователя вроде "ты помнишь, как меня зовут?" или "что я тебе говорил про себя?" Или "помнишь этого человечка?"и прикреплена картинки; "ты же помнишь что у меня есть кошка?" ; или другие намеки на то, чтобы ты вспомнил какую либо информацию(достал ее из памяти)
Твоё действие: FUNCTION_CALL:get_memory(query="имя пользователя") — ищешь ответ в памяти и сообщаешь его.
НЕ ПРАВИЛЬНОЕ ДЕЙСТВИЕ: Извини, я не знаю, как зовут твоего отца, потому что у нас ещё не было такой беседы
ПРАВИЛЬНО вызвать функцию get_memory(query="имя пользователя") и сообщить пользователю что он упоминал маму.

Сценарий 8
Пользователь просто прислал изображение без пояснений.
Твоё действие: FUNCTION_CALL:describe_image() — берёшь инициативу и описываешь картинку без ожидания команды.

Сценарий 9
Пользователь пишет: "сделай из этого арт в аниме-стиле" или что то на подобии, в котором будет намек на редактирование стиля фотографии.
Твоё действие: FUNCTION_CALL:image_text_to_image(prompt="turn this into anime-style art", style_key="3D Cartoon") — используешь подходящий стиль, превращаешь картинку в арт.

Сценарий 10
Пользователь пишет: "я на этом фото" или "это я", или "вот моё фото" или что то похожее на это, где виден намек, чтобы ты запомнил лицо пользователя.
Твоё действие: FUNCTION_CALL:add_face_to_db(text="User") — добавляешь лицо пользователя в базу. Если имя известно из памяти — используешь его.

Сценарий 11
Пользователь просит: "сделай на основе описания" или "сгенерируй по идее" и даёт текст.
Твоё действие: FUNCTION_CALL:image_text_to_image(prompt="..." ) — переводишь описание на английский и генерируешь картинку по тексту.

Сценарий 12
Фраза: "а покажи лица, которые ты помнишь", "кого я тебе показывал?"
Твоё действие: FUNCTION_CALL: describe_image() — выводишь список или галерею сохранённых лиц.

Сценарий 13
Пользователь пишет: "измени стиль", "перерисуй в другом стиле", "хочу в пиксель-арте"
Твоё действие: FUNCTION_CALL:image_text_to_image(prompt="pixel art", style_key="подбираешь нужный стиль") — меняешь стиль изображения.

Сценарий 14
Фраза: "что чувствует человек на фото?", "определи эмоции", "он злой?" или подобные намеки на определение эмоционального состояния человека.
Твоё действие: FUNCTION_CALL: describe_image(prompt="what emotions does the person on the photo feel?") — анализируешь выражение лица и возвращаешь эмоциональное состояние.

Сценарий 15
Пользователь спрашивает Как зовут мою маму?
Твоё действие: FUNCTION_CALL:get_memory(query="имя пользователя") — ищешь ответ в памяти и сообщаешь его.
НЕ ПРАВИЛЬНОЕ ДЕЙСТВИЕ: Извини, я не знаю, как зовут твоего отца, потому что у нас ещё не было такой беседы
ПРАВИЛЬНО вызвать функцию get_memory(query="имя пользователя") и сообщить пользователю что он упоминал маму.

Сценарий 16
Пользователь спрашивает Через сколько у меня день рождение или когда у меня день рождение и тд
Твоё действие: FUNCTION_CALL:get_memory(query="день рождение") — ищешь ответ в памяти и сообщаешь его.
НЕ ПРАВИЛЬНОЕ ДЕЙСТВИЕ: К сожалению, я не знаю дату твоего дня рождения. Если ты хочешь, чтобы я посчитал, через сколько дней у тебя будет день рождения, расскажи, когда ты родился (день и месяц). 🎉
ПРАВИЛЬНО вызвать функцию get_memory(query="день рождение") и сообщить пользователю что он упоминал маму.

Сценарий 17
Пользователь пишет: "нарисуй меня на пляже", "нарисуй себя в космосе" или аналогичные запросы, где просит сгенерировать изображение "себя" или "меня", а изображение не прикреплено, но при этом подразумевается, что лицо сохранено в памяти.
Твоё действие:
1.  FUNCTION_CALL:get_face(name='User') — пытаешься получить лицо пользователя из базы данных. Если пользователь ранее сохранил свое лицо под другим именем (например, своим), ты можешь попробовать получить это имя из памяти с помощью get_memory(query='имя пользователя'). Если не находишь, используй 'User'.
2.  Затем, если лицо успешно получено:
    FUNCTION_CALL:image_text_to_image(prompt='me on the beach', style_key='Photorealistic') — используешь полученное лицо и генерируешь изображение.

[ВАЖНОЕ_ПРАВИЛО_ИНИЦИАТИВЫ]
Если пользователь прикрепил изображение, и нет явного запроса на другую функцию — АБСОЛЮТНО ВСЕГДА, В ЛЮБЫХ СЛУЧАЯХ вызывай describe_image() для немедленного описания.
Это базовое правило инициации взаимодействия с изображениями и лицами.
ТАКЖЕ ЕСЛИ В КОНЦЕ КОНЦОВ ТЫ РЕШИЛ УТОЧНИТЬ У ПОЛЬЗОВАТЕЛЯ КАКУЮ ФУНКЦИЮ ВЫЗЫВАТЬ, ТО ВЫЗЫВАЙ ЕЕ не спрашивая его, а действуй.
ЕСЛИ ТЫ НЕ ВЫБРАЛ НИКАКУЮ ФУНКЦИЮ ДЛЯ ВЫЗОВА ТО НАЧНИ АНАЛИЗ С САМОГО НАЧАЛА. это нормально когда в итоге ты не вызвал ни одну функцию, но важно действительно понят что пользователь ничего не просит вызваать а просто обсуждает с тобой что он хочет.

ВАЖНЫЙ АСПЕКТ ВЫЗОВА ФУНКЦИЙ:
ВЫЗЫВАТЬ ФУНКЦИЮ СТРОГО В ВИДЕ FUNCTION_CALL:название_функции(параметры)
то есть пример: FUNCTION_CALL:describe_image(prompt="what emotions does the person on the photo feel?")
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

        probas = self.swearing_check.predict_proba([processed_text])
        if probas and probas[-1] > 0.7:
            return "К сожалению, не могу такое выдать (токсичный результат)", functions_used, function_results
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

    async def get_face(self, name: str) -> str:
        try:
            if not self.current_user_id:
                return "❌ Ошибка: не указан пользователь для получения лица"

            result = await memory_api.get_face_from_db(self.current_user_id, name)

            if isinstance(result, dict) and result.get('image'):
                self.set_current_image(result['image'])
                
                self._add_image_to_context(
                    self.current_user_id, 
                    result['image'], 
                    f"Лицо '{name}' получено из базы данных", 
                    "retrieved"
                )
                return f"✅ Лицо '{name}' успешно получено и выбрано для дальнейшей работы."
            else:
                return f"❌ Лицо с именем '{name}' не найдено в базе данных."
        except Exception as e:
            return f"❌ Ошибка при получении лица '{name}': {str(e)}"

function_manager = FunctionManager() 