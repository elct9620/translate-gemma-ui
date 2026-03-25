import logging
from collections.abc import Iterator
from threading import Thread
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)

_QUANTIZATION_VRAM_THRESHOLD = 8 * 1024**3  # 8 GB


def _is_model_cached(model_id: str) -> bool:
    """Check if a HuggingFace model is available in the local cache."""
    from huggingface_hub import try_to_load_from_cache

    result = try_to_load_from_cache(model_id, "config.json")
    return isinstance(result, str)


class OutOfMemoryError(RuntimeError):
    """Raised when GPU runs out of memory during model inference."""

    pass


def _is_oom_error(exc: BaseException) -> bool:
    """Check if an exception is a CUDA/MPS out-of-memory error."""
    try:
        import torch

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except ImportError:
        pass
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()

SUPPORTED_LANGUAGES: dict[str, str] = {
    "en": "English",
    "zh-TW": "Chinese (Traditional)",
    "ja": "Japanese",
}


@runtime_checkable
class Translator(Protocol):
    @property
    def languages(self) -> dict[str, str]:
        """Language code -> display name mapping."""
        ...

    @property
    def max_tokens(self) -> int:
        """Max input tokens for the model."""
        ...

    @property
    def is_ready(self) -> bool:
        """Whether the model is loaded."""
        ...

    @property
    def is_quantized(self) -> bool:
        """Whether the model was loaded with quantization."""
        ...

    @property
    def model_name(self) -> str:
        """Display name identifying this translator."""
        ...

    def translate(self, text: str, source_lang: str, target_lang: str) -> Iterator[str]:
        """Yield progressively accumulated translation text (streaming)."""
        ...


class TranslateGemmaTranslator:
    def __init__(
        self,
        model_id: str = "google/translategemma-4b-it",
        token: str | None = None,
        vram_bytes: int | None = None,
    ):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        logger.info("Loading model %s...", model_id)

        self._model_name = model_id
        cached = _is_model_cached(model_id)
        if cached:
            logger.info("Model found in local cache, loading offline")

        resolved_token = None if cached else token

        self._processor = AutoProcessor.from_pretrained(
            model_id, token=resolved_token, local_files_only=cached
        )
        self._languages = SUPPORTED_LANGUAGES

        dtype = torch.bfloat16 if torch.cuda.is_available() or torch.backends.mps.is_available() else torch.float32

        quantization_config = None
        self._is_quantized = False

        if vram_bytes is not None and vram_bytes < _QUANTIZATION_VRAM_THRESHOLD:
            try:
                from transformers import BitsAndBytesConfig

                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                )
                self._is_quantized = True
                logger.info("VRAM (%.2f GB) below threshold; enabling 4-bit quantization", vram_bytes / (1024**3))
            except ImportError:
                logger.warning("bitsandbytes not installed; loading without quantization (may OOM)")

        load_kwargs: dict = {"device_map": "auto", "token": resolved_token, "local_files_only": cached}
        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config
        else:
            load_kwargs["dtype"] = dtype

        self._model = AutoModelForImageTextToText.from_pretrained(model_id, **load_kwargs)
        self._max_tokens = getattr(self._model.config, "max_position_embeddings", 8192)
        self._is_ready = True
        logger.info("Model loaded successfully on %s", self._model.device)

    @property
    def languages(self) -> dict[str, str]:
        return self._languages

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    @property
    def is_quantized(self) -> bool:
        return self._is_quantized

    @property
    def model_name(self) -> str:
        return self._model_name

    def translate(self, text: str, source_lang: str, target_lang: str) -> Iterator[str]:
        from transformers import TextIteratorStreamer

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "source_lang_code": source_lang,
                        "target_lang_code": target_lang,
                        "text": text,
                    }
                ],
            }
        ]
        inputs = self._processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        ).to(self._model.device)

        logger.debug("Chat template applied for: %s -> %s", source_lang, target_lang)

        streamer = TextIteratorStreamer(
            self._processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        generation_kwargs = {
            **inputs,
            "max_new_tokens": 2048,
            "do_sample": False,
            "streamer": streamer,
        }

        thread = Thread(target=self._model.generate, kwargs=generation_kwargs)
        thread.start()

        try:
            accumulated = ""
            for chunk in streamer:
                accumulated += chunk
                yield accumulated

            thread.join()
        except Exception as e:
            thread.join()
            if _is_oom_error(e):
                raise OutOfMemoryError("記憶體不足，建議關閉其他應用程式或改用 CPU 模式") from e
            raise


class FakeTranslator:
    def __init__(self):
        self._languages = SUPPORTED_LANGUAGES

    @property
    def languages(self) -> dict[str, str]:
        return self._languages

    @property
    def max_tokens(self) -> int:
        return 1024

    @property
    def is_ready(self) -> bool:
        return True

    @property
    def is_quantized(self) -> bool:
        return False

    @property
    def model_name(self) -> str:
        return "FakeTranslator"

    def translate(self, text: str, source_lang: str, target_lang: str) -> Iterator[str]:
        target_name = self._languages.get(target_lang, target_lang)
        result = f"[{target_name}] {text}"
        accumulated = ""
        for char in result:
            accumulated += char
            yield accumulated
