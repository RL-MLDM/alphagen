from logging import Logger
import sys
from typing import Optional

from .base import ChatClient, ChatConfig


class ReplChatClient(ChatClient):
    def __init__(self, logger: Optional[Logger] = None):
        super().__init__(ChatConfig(logger=logger))

    def chat_complete(self, content: str) -> str:
        self._add_message("user", content)
        print(f'{"=" * 28}QUERY{"=" * 28}')
        print(content)
        print(f'{"=" * 20}INPUT LLM ANSWER HERE{"=" * 20}')
        answer = "".join(sys.stdin.readlines()).rstrip('\n')
        self._add_message("assistant", answer)
        return answer
