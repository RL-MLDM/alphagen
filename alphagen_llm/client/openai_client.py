from dataclasses import asdict
from openai import OpenAI
import tokentrim as tt
from tokentrim.model_map import MODEL_MAX_TOKENS

from .base import ChatClient, ChatConfig


class OpenAIClient(ChatClient):
    def __init__(
        self,
        client: OpenAI,
        config: ChatConfig,
        model: str = "gpt-3.5-turbo-0125",
        trim_to_token_limit: bool = True
    ) -> None:
        super().__init__(config)
        _update_model_max_tokens()
        self._client = client
        self._model = model
        self._trim = trim_to_token_limit

    def chat_complete(self, content: str) -> str:
        self._add_message("user", content)
        idx = int(self._system_prompt is not None)
        messages = [asdict(msg) for msg in self._dialog[idx:]]
        response = self._client.chat.completions.create(
            model=self._model,
            messages=tt.trim(messages, self._model, self._system_prompt)    # type: ignore
        )
        result: str = response.choices[0].message.content       # type: ignore
        self._add_message("assistant", result)
        return result

    def _on_reset(self) -> None:
        self._start_idx = 0

    _client: OpenAI
    _model: str


_UPDATED = False


def _update_model_max_tokens():
    global _UPDATED
    if _UPDATED:
        return
    MODEL_MAX_TOKENS["gpt-3.5-turbo"] = 16385
    MODEL_MAX_TOKENS["gpt-3.5-turbo-1106"] = 16385
    MODEL_MAX_TOKENS["gpt-3.5-turbo-0125"] = 16385
