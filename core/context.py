
import json
import time
import logging
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
import aiofiles

logger = logging.getLogger(__name__)

@dataclass
class Message:
    role: str             
    content: str           
    timestamp: float      
    function_calls: List[str] = None   
    function_results: Dict = None      
    
    def __post_init__(self):
        if self.function_calls is None:
            self.function_calls = []
        if self.function_results is None:
            self.function_results = {}

@dataclass
class UserContext:  
    user_id: str
    role: str
    messages: List[Message]
    created_at: float
    last_activity: float
    
    def add_message(self, message: Message):
        self.messages.append(message)
        self.last_activity = time.time()
    
    def get_conversation_history(self) -> List[Dict]:
        return [
            {
                "role": msg.role,
                "content": msg.content
            }
            for msg in self.messages
        ]

class ContextManager:
    def __init__(self, storage_path: str = "user_contexts"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self.contexts: Dict[str, UserContext] = {}
        
        logger.info(f"📁 Менеджер контекстов инициализирован: {self.storage_path}")
    
    def _get_context_key(self, user_id: str, role: str) -> str:
        return f"{user_id}_{role}"
    
    def _get_file_path(self, user_id: str, role: str) -> Path:
        return self.storage_path / f"{user_id}_{role}.json"
    
    async def get_or_create_context(self, user_id: str, role: str) -> UserContext:
        context_key = self._get_context_key(user_id, role)
        
        if context_key in self.contexts:
            return self.contexts[context_key]
        
        context = await self._load_context(user_id, role)
        if context:
            self.contexts[context_key] = context
            logger.info(f"📂 Загружен контекст {context_key} ({len(context.messages)} сообщений)")
            return context
        
        context = UserContext(
            user_id=user_id,
            role=role,
            messages=[],
            created_at=time.time(),
            last_activity=time.time()
        )
        
        self.contexts[context_key] = context
        await self._save_context(context)
        
        logger.info(f"🆕 Создан новый контекст {context_key}")
        return context
    
    async def add_user_message(self, user_id: str, content: str, role: str) -> UserContext:
        context = await self.get_or_create_context(user_id, role)
        
        message = Message(
            role="user",
            content=content,
            timestamp=time.time()
        )
        
        context.add_message(message)
        await self._save_context(context)
        
        return context
    
    async def add_assistant_message(self, user_id: str, content: str, role: str,
                            function_calls: List[str] = None, 
                            function_results: Dict = None) -> UserContext:
        context = await self.get_or_create_context(user_id, role)
        
        message = Message(
            role="assistant",
            content=content,
            timestamp=time.time(),
            function_calls=function_calls or [],
            function_results=function_results or {}
        )
        
        context.add_message(message)
        await self._save_context(context)
        
        return context
    
    async def add_system_message(self, user_id: str, content: str, role: str) -> UserContext:
        context = await self.get_or_create_context(user_id, role)
        
        message = Message(
            role="system",
            content=content,
            timestamp=time.time()
        )
        
        context.add_message(message)
        await self._save_context(context)
        
        return context
    
    async def add_or_update_system_message(self, user_id: str, content: str, role: str) -> UserContext:
        context = await self.get_or_create_context(user_id, role)
        
        system_message = Message(
            role="system",
            content=content,
            timestamp=time.time()
        )
        
        if context.messages and context.messages[0].role == "system":
            context.messages[0] = system_message
        else:
            context.messages.insert(0, system_message)
        
        context.last_activity = time.time()
        await self._save_context(context)
        
        return context
    
    async def clear_context(self, user_id: str, role: str):
        context_key = self._get_context_key(user_id, role)

        if context_key in self.contexts:
            self.contexts.pop(context_key, None)
  
        file_path = self._get_file_path(user_id, role)
        try:
            if file_path.exists():
                file_path.unlink()
                logger.info(f"🗑️ Файл контекста {file_path} удалён")
            else:
                logger.info(f"🗑️ Файл контекста {file_path} не найден для удаления")
        except Exception as e:
            logger.error(f"❌ Ошибка при удалении файла контекста {file_path}: {e}")
        logger.info(f"🗑️ Очищен контекст {context_key}")
    
    async def get_context_info(self, user_id: str, role: str = None) -> Optional[Dict]:
        if role is None:    
            for key, context in self.contexts.items():
                if context.user_id == user_id:
                    return {
                        "user_id": context.user_id,
                        "role": context.role,
                        "message_count": len(context.messages),
                        "created_at": datetime.fromtimestamp(context.created_at).isoformat(),
                        "last_activity": datetime.fromtimestamp(context.last_activity).isoformat()
                    }
            return None
        
        context_key = self._get_context_key(user_id, role)
        context = self.contexts.get(context_key)
        
        if not context:
            context = await self._load_context(user_id, role)
            if not context:
                return None
        
        return {
            "user_id": context.user_id,
            "role": context.role,
            "message_count": len(context.messages),
            "created_at": datetime.fromtimestamp(context.created_at).isoformat(),
            "last_activity": datetime.fromtimestamp(context.last_activity).isoformat()
        }
    
    async def _save_context(self, context: UserContext):
        try:
            file_path = self._get_file_path(context.user_id, context.role)
            
            context_data = {
                "user_id": context.user_id,
                "role": context.role,
                "created_at": context.created_at,
                "last_activity": context.last_activity,
                "messages": [
                    {
                        "role": msg.role,
                        "content": msg.content,
                        "timestamp": msg.timestamp,
                        "function_calls": msg.function_calls,
                        "function_results": msg.function_results
                    }
                    for msg in context.messages
                ]
            }
            
            async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(context_data, ensure_ascii=False, indent=2))
                
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения контекста {context.user_id}_{context.role}: {e}")
    
    async def _load_context(self, user_id: str, role: str) -> Optional[UserContext]:
        try:
            file_path = self._get_file_path(user_id, role)
            
            if not file_path.exists():
                return None
            
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
                data = json.loads(content)
            
            messages = [
                Message(
                    role=msg_data["role"],
                    content=msg_data["content"],
                    timestamp=msg_data["timestamp"],
                    function_calls=msg_data.get("function_calls", []),
                    function_results=msg_data.get("function_results", {})
                )
                for msg_data in data["messages"]
            ]
            
            return UserContext(
                user_id=data["user_id"],
                role=data["role"],
                messages=messages,
                created_at=data["created_at"],
                last_activity=data["last_activity"]
            )
            
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки контекста {user_id}_{role}: {e}")
            return None

context_manager = ContextManager() 