from __future__ import annotations

import os
from dataclasses import dataclass


def _optional_int(value: str | None) -> int | None:
    if value in (None, "", "none", "null"):
        return None
    return int(value)


@dataclass(frozen=True)
class Settings:
    diagnosis_backend: str = os.getenv("PSYCHE_DIAGNOSIS_BACKEND", "rule")
    chatglm_model_id: str = os.getenv("PSYCHE_CHATGLM_MODEL_ID", "THUDM/chatglm2-6b")
    chatglm_revision: str | None = os.getenv("PSYCHE_CHATGLM_REVISION") or None
    chatglm_device: str = os.getenv("PSYCHE_CHATGLM_DEVICE", "cuda")
    chatglm_quantization_bits: int | None = _optional_int(os.getenv("PSYCHE_CHATGLM_QUANTIZATION_BITS", "4"))


def get_settings() -> Settings:
    return Settings()
