import logging
import os
from typing import List, Dict, Any

from dotenv import load_dotenv
load_dotenv()

import aiohttp
import asyncio

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

VLLM_API_URL = "http://llm-model:8000"
VLLM_MODEL = "/model" 
VLLM_TIMEOUT = 120

# vlm
VLM_API_URL =  "http://vlm:8001"

# gen
GEN_API_URL =  "http://image-gen:8003"

# memory    
MEMORY_API_URL = "http://ltm-api:8006"


class VLLMClient:
    def __init__(self, api_url: str, model: str, timeout: int = 120):
        self.api_url = api_url
        self.model = model
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        logger.info(f"🚀 vLLM клиент: {api_url}, модель: {model}")
    
    async def generate(self, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 2048),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.95),
            "top_k": kwargs.get("top_k", 20),
            "min_p": kwargs.get("min_p", 0.0),
            "presence_penalty": kwargs.get("presence_penalty", 0.0),
            "repetition_penalty": kwargs.get("repetition_penalty", 1.1),
            "stream": False
        }
        
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.post(
                    f"{self.api_url}/v1/chat/completions",
                    json=payload,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        print(result)
                        return {
                            "success": True,
                            "result": result,
                            "model": result.get("model", self.model),
                            "usage": result.get("usage", {})
                        }
                    else:
                        error = await response.text()
                        logger.error(f"❌ vLLM API error: {response.status} - {error}")
                        return {
                            "success": False,
                            "error": f"API error: {response.status}",
                            "details": error
                        }
        except asyncio.TimeoutError:
            logger.error("❌ vLLM API timeout")
            return {
                "success": False,
                "error": "Request timeout"
            }
        except Exception as e:
            logger.error(f"❌ vLLM API exception: {e}")
            return {
                "success": False,
                "error": "Connection error",
                "details": str(e)
            }
    
    async def health_check(self) -> bool:
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                async with session.get(f"{self.api_url}/health") as response:
                    return response.status == 200
        except:
            return False
        
class VLMClient:
    def __init__(self, api_url: str, timeout: int = 120):
        self.api_url = api_url
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        logger.info(f"🚀 vlm клиент: {api_url}")

    async def describe_image(self, image_bs64: str, prompt: str = "") -> str:
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.post(
                    f"{self.api_url}/analyze",
                    json={"image_base64": image_bs64, "prompt": prompt}
                ) as response:
                    return await response.text()
        except:
            return 'Произошла ошибка при описании изображения'

    async def save_face_to_db(self, face_bs64: str, text: str) -> str:
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.post(
                    f"{self.api_url}/add",
                    json={"image_base64": face_bs64, "comment": text}
                ) as response:
                    return await response.text()
        except:
            return 'Произошла ошибка при сохранении изображения лица в базу данных'

class GenAPI:
    def __init__(self, api_url: str, timeout: int = 120):
        self.api_url = api_url
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        logger.info(f"🚀 gen api: {api_url}")
    
    async def prompt_to_image(self, prompt: str, style_key: str) -> Dict[str, Any]:
        payload = {"prompt": prompt, "style_key": style_key}

        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.post(
                    f"{self.api_url}/generate_from_text",
                    json=payload
                ) as response:
                    return await response.json()
        except:
            return 'Произошла ошибка при генерации изображения'
        
    async def image_to_image(self, image_bs64: str, prompt: str, style_key: str) -> Dict[str, Any]:
        payload = {"image": image_bs64, "prompt": prompt, "style_key": style_key}   

        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.post(
                    f"{self.api_url}/generate_from_image_text",
                    json=payload
                ) as response:
                    return await response.json()
        except:
            return 'Произошла ошибка при генерации изображения'
        
    async def sketch_to_image(self, image_bs64: str, prompt: str) -> Dict[str, Any]:
        payload = {"image": image_bs64, "prompt": prompt}

        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.post(f"{self.api_url}/sketch_to_image", json=payload) as response:
                    return await response.json()
        except:
            return 'Произошла ошибка при генерации изображения'

    async def create_avatar(self, prompt: str) -> Dict[str, Any]:
        payload = {"prompt": prompt + "A cute 3D character portrait in Pixar Disney style, soft lighting, big expressive eyes, friendly smile, pastel colors, upper body shot, studio background", "style_key": ""}

        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.post(f"{self.api_url}/generate_from_text", json=payload) as response:
                    return await response.json()
        except:
            return 'Произошла ошибка при генерации изображения'
        
        
class MemoryAPI:
    def __init__(self, api_url: str, timeout: int = 120):
        self.api_url = api_url
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        logger.info(f"🚀 memory api: {api_url}")

    async def save_memory(self, user_id: str, content: str, context: str = None) -> Dict[str, Any]:
        try:
            print(f"🔧 Сохраняем память: {content}, {context}, {user_id}, {self.api_url}")
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.post(f"{self.api_url}/memory/store", json={"content": content, "context": context, "user_id": user_id}) as response:
                    return await response.json()
        except:
            return 'Произошла ошибка при получении памяти'
        
    async def get_memory(self, user_id: str, query: str) -> Dict[str, Any]:
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.post(f"{self.api_url}/search", json={"query": query, "user_id": user_id}) as response:
                    return await response.json()
        except:
            return 'Произошла ошибка при сохранении памяти'
        
vllm_client = VLLMClient(VLLM_API_URL, VLLM_MODEL, VLLM_TIMEOUT)

vlm_client = VLMClient(VLM_API_URL, VLLM_TIMEOUT)

gen_api = GenAPI(GEN_API_URL, VLLM_TIMEOUT)

memory_api = MemoryAPI(MEMORY_API_URL, VLLM_TIMEOUT)