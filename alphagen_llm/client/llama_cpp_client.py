from typing import Optional, List
import requests as req
from .base import ChatClient, ChatConfig, Message


_B_INST, _E_INST = "[INST]", "[/INST]"
_B_SYS, _E_SYS = "<<SYS>>\n", "\n<<SYS>>\n\n"
_SPECIAL_TAGS = [_B_INST, _E_INST, _B_SYS, _E_SYS]
_BOS_ID, _EOS_ID = 1, 2             # TODO: Hardcoded Llama 2 token ID


class LlamaCppClient(ChatClient):
    def __init__(
        self,
        endpoint: str,
        config: Optional[ChatConfig] = None
    ) -> None:
        super().__init__(config)
        if self._system_prompt is not None:
            self._ensure_no_special_tags(self._system_prompt)
        self._endpoint = endpoint
        self._slot_id = -1

    def chat_complete(self, content: str) -> str:
        self._ensure_no_special_tags(content)
        self._add_message("user", content)
        res = req.post(f"{self._endpoint}/completion", json={
            "prompt": self._prompt(),
            "slot_id": self._slot_id
        })
        obj = res.json()
        self._slot_id = obj["slot_id"]
        answer: str = obj["content"].strip()
        self._add_message("assistant", answer)
        return answer

    def tokenize(self, prompt: str, bos: bool = False, eos: bool = False) -> List[int]:
        res = req.post(f"{self._endpoint}/tokenize", json={"content": prompt})
        ids: list[int] = res.json()["tokens"]
        if bos:
            ids.insert(0, _BOS_ID)
        if eos:
            ids.append(_EOS_ID)
        return ids

    def decode(self, token_ids: List[int]) -> str:
        res = req.post(f"{self._endpoint}/detokenize", json={"tokens": token_ids})
        return res.json()["content"]

    _endpoint: str
    _slot_id: int

    def _prompt(self) -> List[int]:
        dialog = self._dialog
        if dialog[0].role == "system":
            assert len(dialog) > 1, "No user prompt after the system prompt."
            dialog = [Message(
                role=dialog[1].role,
                content=f"{_B_SYS}{dialog[0].content}{_E_SYS}{dialog[1].content}"
            )] + dialog[2:]
        assert (
            len(dialog) % 2 == 1 and
            all(msg.role == "user" for msg in dialog[0::2]) and
            all(msg.role == "assistant" for msg in dialog[1::2])
        ), (
            "The roles in the dialog must be user/assistant alternating, "
            "with an optional system prompt in the front."
        )
        tokens = sum(
            (self._tokenize_qa_pair(q, a) for q, a in zip(dialog[0:-1:2], dialog[1::2])),
            []
        )
        tokens += self._tokenize_qa_pair(dialog[-1])
        return tokens

    def _tokenize_qa_pair(self, prompt: Message, answer: Optional[Message] = None) -> List[int]:
        prompt_str = f"{_B_INST} {prompt.content.strip()} {_E_INST}"
        if answer is None:
            return self.tokenize(prompt_str, bos=True, eos=False)
        prompt_str += f" {answer.content.strip()} "
        return self.tokenize(prompt_str, bos=True, eos=True)

    @classmethod
    def _ensure_no_special_tags(cls, prompt: str) -> None:
        assert not any(tag in prompt for tag in _SPECIAL_TAGS), "The message contains special tag."
