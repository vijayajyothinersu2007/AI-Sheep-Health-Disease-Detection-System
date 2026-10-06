"""
LLM Client: Integrates with Google Gemini SDK / REST API with robust retries,
and provides a safe local grounded synthesis fallback when API keys are not set.
"""
import os
import json
import logging
import time
from typing import Optional, Dict, Any, List

from .config import (
    GEMINI_API_KEY,
    DEFAULT_GEMINI_MODEL,
    OPENAI_API_KEY,
    GROQ_API_KEY,
    DEFAULT_OPENAI_MODEL
)
from .prompts import VETERINARY_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Unified LLM Client supporting Google Gemini, OpenAI, Groq, with an intelligent
    grounded knowledge synthesizer fallback when offline or without API keys.
    """
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        # Refresh env keys dynamically
        self.gemini_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or GEMINI_API_KEY
        self.openai_key = os.environ.get("OPENAI_API_KEY") or OPENAI_API_KEY
        self.groq_key = os.environ.get("GROQ_API_KEY") or GROQ_API_KEY
        self.gemini_model = model or DEFAULT_GEMINI_MODEL
        
        # Initialize Google GenAI client if available and key is valid
        self._genai_client = None
        if self.gemini_key and not self.gemini_key.startswith("hf_"):
            try:
                from google import genai
                self._genai_client = genai.Client(api_key=self.gemini_key)
            except Exception as e:
                logger.warning(f"Could not initialize google.genai Client: {e}")
                self._genai_client = None
        elif self.gemini_key and self.gemini_key.startswith("hf_"):
            logger.info("Notice: GEMINI_API_KEY has 'hf_' prefix. Running in Grounded Veterinary Knowledge Engine mode.")

    def is_api_configured(self) -> bool:
        """Returns True if any supported remote LLM API key is present and valid."""
        valid_gemini = bool(self.gemini_key and not self.gemini_key.startswith("hf_") and len(self.gemini_key) > 10)
        valid_openai = bool(self.openai_key and len(self.openai_key) > 10)
        valid_groq = bool(self.groq_key and len(self.groq_key) > 10)
        return valid_gemini or valid_openai or valid_groq

    def get_active_provider(self) -> str:
        """Identifies active LLM provider name."""
        if self.gemini_key and not self.gemini_key.startswith("hf_"):
            return f"Google Gemini ({self.gemini_model})"
        elif self.openai_key:
            return f"OpenAI ({DEFAULT_OPENAI_MODEL})"
        elif self.groq_key:
            return "Groq (Llama-3.1)"
        return "Grounded Veterinary Knowledge Engine (Local FAISS Grounding)"

    def generate(
        self,
        prompt: str,
        system_prompt: str = VETERINARY_SYSTEM_PROMPT,
        temperature: float = 0.2,
        max_tokens: int = 2500
    ) -> str:
        """
        Sends prompt to configured LLM or returns gracefully handled grounded fallback.
        """
        # 1. Try Google Gemini SDK or REST
        if self.gemini_key:
            try:
                response = self._call_gemini(prompt, system_prompt, temperature, max_tokens)
                if response and response.strip():
                    return response
            except Exception as e:
                logger.warning(f"Gemini API call failed: {e}")

        # 2. Try OpenAI if configured
        if self.openai_key:
            try:
                response = self._call_openai(prompt, system_prompt, temperature, max_tokens)
                if response and response.strip():
                    return response
            except Exception as e:
                logger.warning(f"OpenAI API call failed: {e}")

        # 3. Try Groq if configured
        if self.groq_key:
            try:
                response = self._call_groq(prompt, system_prompt, temperature, max_tokens)
                if response and response.strip():
                    return response
            except Exception as e:
                logger.warning(f"Groq API call failed: {e}")

        return ""

    def _call_gemini(self, prompt: str, system_prompt: str, temp: float, max_tokens: int) -> str:
        """Invokes Google Gemini via google.genai SDK or REST API."""
        if not self.gemini_key or self.gemini_key.startswith("hf_"):
            return ""
        full_text = f"{system_prompt}\n\nUser Request:\n{prompt}"
        models_to_try = [self.gemini_model, "gemini-2.5-flash", "gemini-1.5-flash", "gemini-1.5-pro"]
        
        # Deduplicate while preserving order
        unique_models = []
        for m in models_to_try:
            if m and m not in unique_models:
                unique_models.append(m)

        # 1. Attempt using google.genai SDK
        if self._genai_client is not None:
            for model_name in unique_models:
                try:
                    response = self._genai_client.models.generate_content(
                        model=model_name,
                        contents=full_text,
                    )
                    if hasattr(response, "text") and response.text:
                        return response.text
                except Exception as ex:
                    logger.debug(f"SDK call failed for {model_name}: {ex}")
                    continue

        # 2. Fallback to Gemini REST API with requests
        import requests
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": full_text}]}],
            "generationConfig": {
                "temperature": temp,
                "maxOutputTokens": max_tokens
            }
        }
        
        for model_name in unique_models:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={self.gemini_key}"
            try:
                res = requests.post(url, headers=headers, json=payload, timeout=30)
                if res.status_code == 200:
                    data = res.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if parts and "text" in parts[0]:
                            return parts[0]["text"]
            except Exception as ex:
                logger.debug(f"REST call failed for {model_name}: {ex}")
                continue

        return ""

    def _call_openai(self, prompt: str, system_prompt: str, temp: float, max_tokens: int) -> str:
        """Invokes OpenAI Chat Completion API."""
        import requests
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.openai_key}"
        }
        payload = {
            "model": DEFAULT_OPENAI_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": temp,
            "max_tokens": max_tokens
        }
        res = requests.post(url, headers=headers, json=payload, timeout=30)
        if res.status_code == 200:
            data = res.json()
            choices = data.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "")
        return ""

    def _call_groq(self, prompt: str, system_prompt: str, temp: float, max_tokens: int) -> str:
        """Invokes Groq Cloud Completion API."""
        import requests
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.groq_key}"
        }
        payload = {
            "model": "llama-3.1-8b-instant",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": temp,
            "max_tokens": max_tokens
        }
        res = requests.post(url, headers=headers, json=payload, timeout=30)
        if res.status_code == 200:
            data = res.json()
            choices = data.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "")
        return ""
