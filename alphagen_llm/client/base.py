from abc import ABCMeta, abstractmethod
from typing import Literal, List, Optional, Callable, Tuple, Union, overload
from dataclasses import dataclass
from logging import Logger

from alphagen.utils.logging import get_null_logger


Role = Literal["system", "user", "assistant"]


@dataclass
class Message:
    role: Role
    content: str


Dialog = List[Message]
MessageFormatter = Callable[[str, str], str]
def _default_msg_fmt(role: str, content: str) -> str: return f"[{role}] {content}"


@dataclass
class ChatConfig:
    system_prompt: Optional[str] = None
    logger: Optional[Logger] = None
    message_formatter: MessageFormatter = _default_msg_fmt


class ChatClient(metaclass=ABCMeta):
    def __init__(self, config: Optional[ChatConfig] = None) -> None:
        config = config or ChatConfig()
        self._system_prompt = config.system_prompt
        self._logger = config.logger or get_null_logger()
        self._msg_fmt = config.message_formatter
        self.reset(self._system_prompt)

    @property
    def dialog(self) -> Dialog: return self._dialog

    @property
    def logger(self) -> Logger: return self._logger

    @property
    def message_formatter(self) -> MessageFormatter: return self._msg_fmt

    @overload
    def log_message(self, msg: Tuple[str, str]) -> None: ...

    @overload
    def log_message(self, msg: Message) -> None: ...

    def log_message(self, msg: Union[Tuple[str, str], Message]) -> None:
        if isinstance(msg, Message):
            self._logger.debug(self._msg_fmt(msg.role, msg.content))
        else:
            self._logger.debug(self._msg_fmt(*msg))

    def reset(self, system_prompt: Optional[str] = None) -> None:
        self.log_message(("script", "Dialog history is reset!"))
        self._dialog = []
        if (sys := system_prompt or self._system_prompt) is not None:
            self._add_message("system", sys, write_log=system_prompt is not None)

    @abstractmethod
    def chat_complete(self, content: str) -> str: ...

    def _add_message(self, role: Role, content: str, write_log: bool = True) -> None:
        msg = Message(role, content)
        self._dialog.append(msg)
        if write_log:
            self.log_message(msg)

    _system_prompt: Optional[str]
    _dialog: Dialog
    _logger: Logger
    _msg_fmt: MessageFormatter
