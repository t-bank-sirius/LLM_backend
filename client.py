import logging
import os
from typing import List, Dict, Any

import aiohttp
import asyncio

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


VLLM_API_URL = os.getenv("VLLM_API_URL", "http://localhost:8000")
VLLM_MODEL = os.getenv("VLLM_MODEL", "matvei_pzh")
VLLM_TIMEOUT = int(os.getenv("VLLM_TIMEOUT", "120"))

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
            "repetition_penalty": kwargs.get("repetition_penalty", 1.1),
            "stream": False,
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

vllm_client = VLLMClient(VLLM_API_URL, VLLM_MODEL, VLLM_TIMEOUT)