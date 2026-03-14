"""
Learning Tools Base
-------------------
Shared LLM calling logic for all four learning tools.
Each tool has its own system prompt and output parser,
but they all use the same underlying OpenAI call.
"""

import logging
from openai import OpenAI
from app.config.settings import settings

logger = logging.getLogger(__name__)


class LearningToolsBase:
    """Base class for all learning tools. Handles the OpenAI call."""

    def __init__(self):
        self._client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self._model = settings.OPENAI_LLM_MODEL

    def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.4,
        max_tokens: int = 2000,
    ) -> str:
        """
        Make a single OpenAI chat completion call.
        Returns the raw text response.
        """
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise RuntimeError(f"AI service error: {e}")
