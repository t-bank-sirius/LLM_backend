import re
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class FunctionResult:
    success: bool
    result: Any
    error: Optional[str] = None
    function_name: str = ""

class FunctionManager:
    def __init__(self):
        self.functions: Dict[str, Callable] = {}
        self._register_builtin_functions()
    
    def _register_builtin_functions(self): 
        def get_current_time() -> str:
            return datetime.now().strftime("%H:%M:%S")
        
        def get_current_date() -> str:
            return datetime.now().strftime("%Y-%m-%d")
        
        def get_datetime() -> str:
            return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        def get_weekday() -> str:
            weekdays = ["понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"]
            return weekdays[datetime.now().weekday()]
        
        self.functions["get_current_time"] = get_current_time
        self.functions["get_current_date"] = get_current_date  
        self.functions["get_datetime"] = get_datetime
        self.functions["get_weekday"] = get_weekday
        
        logger.info(f"✅ Зарегистрировано {len(self.functions)} функций")
    
    def get_system_prompt_addition(self) -> str:
        return """

        🔧 ДОСТУПНЫЕ ФУНКЦИИ:
        - get_current_time() - получить текущее время  
        - get_current_date() - получить текущую дату
        - get_datetime() - получить дату и время
        - get_weekday() - получить день недели

        📋 ПРАВИЛА ИСПОЛЬЗОВАНИЯ ФУНКЦИЙ:
        1. Формат вызова: FUNCTION_CALL:имя_функции()
        2. Пример: FUNCTION_CALL:get_current_time()
        3. Ты можешь вызвать несколько функций в одном ответе
        4. После вызова функций ты получишь их результаты и сможешь продолжить ответ
        5. Ты можешь использовать результаты функций для дальнейших вычислений и ответов

        ВАЖНО: Вызывай функции только когда это действительно необходимо для ответа пользователю."""
    
    def parse_and_execute(self, text: str) -> tuple[str, List[str], Dict]:
        pattern = r'FUNCTION_CALL:(\w+)\(\)'
        matches = re.findall(pattern, text)

        functions_used = []
        function_results = {}
        results_output = []

        if not matches:
            return "", [], {}

        logger.info(f"🔧 Найдено {len(matches)} вызовов функций: {matches}")

        for function_name in matches:
            if function_name in self.functions:
                try:
                    result = self.functions[function_name]()
                    functions_used.append(function_name)
                    function_results[function_name] = result
                    results_output.append(f"✅ {function_name}(): {result}")
                    logger.info(f"✅ Выполнена функция {function_name}: {result}")
                except Exception as e:
                    logger.error(f"❌ Ошибка функции {function_name}: {e}")
                    results_output.append(f"❌ {function_name}(): Ошибка - {e}")
                    function_results[function_name] = f"Ошибка: {e}"
            else:
                logger.warning(f"⚠️ Неизвестная функция: {function_name}")
                results_output.append(f"⚠️ {function_name}(): Неизвестная функция")
                function_results[function_name] = "Неизвестная функция"

        processed_text = "\n".join(results_output)
        return processed_text, functions_used, function_results
    
    def is_need_functions(self, text: str) -> bool:
        pattern = r'FUNCTION_CALL:(\w+)\(\)'
        matches = re.findall(pattern, text)
        return len(matches) > 0
    
    def remove_function_calls(self, text: str) -> str:
        pattern = r'FUNCTION_CALL:(\w+)\(\)'
        return re.sub(pattern, '', text).strip()

function_manager = FunctionManager() 