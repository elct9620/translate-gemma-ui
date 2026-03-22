import logging
import re
from collections.abc import Iterator
from threading import Thread
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


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

    def translate(self, text: str, source_lang: str, target_lang: str) -> Iterator[str]:
        """Yield progressively accumulated translation text (streaming)."""
        ...


def _extract_languages_from_template(chat_template: str) -> dict[str, str]:
    pattern = re.compile(r'"([a-zA-Z\-]+)":\s*"([^"]+)"')
    languages: dict[str, str] = {}
    for match in pattern.finditer(chat_template):
        code, name = match.group(1), match.group(2)
        if code not in languages:
            languages[code] = name
    return languages


class TranslateGemmaTranslator:
    def __init__(self, model_id: str = "google/translategemma-4b-it"):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        logger.info("Loading model %s...", model_id)

        self._processor = AutoProcessor.from_pretrained(model_id)
        self._languages = _extract_languages_from_template(self._processor.chat_template)

        dtype = torch.bfloat16 if torch.cuda.is_available() or torch.backends.mps.is_available() else torch.float32
        self._model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=dtype,
        )
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
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device)

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

        accumulated = ""
        for chunk in streamer:
            accumulated += chunk
            yield accumulated

        thread.join()


class FakeTranslator:
    def __init__(self):
        self._languages = {
            "en": "English",
            "zh-TW": "Chinese (Traditional)",
            "zh-CN": "Chinese (Simplified)",
            "ja": "Japanese",
            "ko": "Korean",
            "es": "Spanish",
            "fr": "French",
            "de": "German",
        }

    @property
    def languages(self) -> dict[str, str]:
        return self._languages

    @property
    def max_tokens(self) -> int:
        return 1024

    @property
    def is_ready(self) -> bool:
        return True

    def translate(self, text: str, source_lang: str, target_lang: str) -> Iterator[str]:
        target_name = self._languages.get(target_lang, target_lang)
        result = f"[{target_name}] {text}"
        accumulated = ""
        for char in result:
            accumulated += char
            yield accumulated
