"""Turns retrieved passages into an answer with Gemini."""
from typing import List, Optional

from fastapi import HTTPException
from google import genai
from google.genai import types

from .library import Hit

INSTRUCTIONS = (
    "You answer questions using only the numbered excerpts you are given. "
    "Cite the excerpts you rely on with their numbers, like [1] or [2][3]. "
    "If the excerpts do not contain the answer, say you could not find it in the documents."
)


def build_prompt(question: str, hits: List[Hit]) -> str:
    excerpts = "\n\n".join(
        f"[{number}] ({hit.filename}, page {hit.page})\n{hit.text}" for number, hit in enumerate(hits, start=1)
    )
    return f"Excerpts:\n\n{excerpts}\n\nQuestion: {question}"


class Answerer:
    """Creates the Gemini client on first use, so uploading works even without an API key."""

    def __init__(self, api_key: Optional[str], model: str):
        self._api_key = api_key
        self._model = model
        self._client = None

    @property
    def configured(self) -> bool:
        return bool(self._api_key)

    def answer(self, question: str, hits: List[Hit]) -> str:
        if not self._api_key:
            raise HTTPException(503, "GEMINI_API_KEY is not set. Set it and restart to ask questions.")
        if self._client is None:
            self._client = genai.Client(api_key=self._api_key)
        response = self._client.models.generate_content(
            model=self._model,
            contents=build_prompt(question, hits),
            config=types.GenerateContentConfig(system_instruction=INSTRUCTIONS),
        )
        return response.text or "The model returned no answer (it may have been blocked)."
