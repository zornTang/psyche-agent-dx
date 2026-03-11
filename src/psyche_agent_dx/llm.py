from __future__ import annotations

from dataclasses import dataclass

from psyche_agent_dx.config import Settings


@dataclass
class ChatGeneration:
    text: str


class ChatModel:
    def generate(self, prompt: str) -> ChatGeneration:
        raise NotImplementedError


class ChatGLMChatModel(ChatModel):
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._tokenizer = None
        self._model = None

    def generate(self, prompt: str) -> ChatGeneration:
        self._ensure_loaded()
        response, _history = self._model.chat(self._tokenizer, prompt, history=[])
        return ChatGeneration(text=response)

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "ChatGLM backend requires optional dependencies. "
                "Install with: pip install -e '.[local-llm]'"
            ) from exc

        load_kwargs: dict[str, object] = {"trust_remote_code": True}
        if self._settings.chatglm_revision:
            load_kwargs["revision"] = self._settings.chatglm_revision

        tokenizer = AutoTokenizer.from_pretrained(
            self._settings.chatglm_model_id,
            **load_kwargs,
        )
        model = AutoModel.from_pretrained(
            self._settings.chatglm_model_id,
            **load_kwargs,
        )

        quantization_bits = self._settings.chatglm_quantization_bits
        if quantization_bits in (4, 8):
            model = model.quantize(quantization_bits)

        device = self._settings.chatglm_device
        if device == "cpu":
            model = model.float()
        else:
            if quantization_bits is None:
                model = model.half()
            model = model.to(device)

        self._tokenizer = tokenizer
        self._model = model.eval()


def build_chat_model(settings: Settings) -> ChatModel | None:
    if settings.diagnosis_backend != "chatglm":
        return None
    return ChatGLMChatModel(settings)
