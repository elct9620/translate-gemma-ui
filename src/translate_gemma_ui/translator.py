import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass
from threading import Thread
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TranslationContext:
    previous: list[str]
    following: list[str]


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
    def model_name(self) -> str:
        """Display name identifying this translator."""
        ...

    def translate(
        self, text: str, source_lang: str, target_lang: str, context: TranslationContext | None = None
    ) -> Iterator[str]:
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
    def __init__(self, model_id: str = "google/translategemma-4b-it", token: str | None = None):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        logger.info("Loading model %s...", model_id)

        self._model_name = model_id
        self._processor = AutoProcessor.from_pretrained(model_id, token=token)
        self._languages = _extract_languages_from_template(self._processor.chat_template)

        dtype = torch.bfloat16 if torch.cuda.is_available() or torch.backends.mps.is_available() else torch.float32
        self._model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            device_map="auto",
            dtype=dtype,
            token=token,
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

    @property
    def model_name(self) -> str:
        return self._model_name

    def _build_context_prompt(
        self, text: str, source_lang: str, target_lang: str, context: TranslationContext
    ) -> str:
        source_name = self._languages.get(source_lang, source_lang)
        target_name = self._languages.get(target_lang, target_lang)

        prompt = (
            f"<start_of_turn>user\n"
            f"You are a professional {source_name} ({source_lang}) to {target_name} ({target_lang}) translator. "
            f"Your goal is to accurately convey the meaning and nuances of the original {source_name} text "
            f"while adhering to {target_name} grammar, vocabulary, and cultural sensitivities.\n"
        )

        context_lines = []
        if context.previous:
            context_lines.append("Previous: " + " ".join(context.previous))
        if context.following:
            context_lines.append("Next: " + " ".join(context.following))

        if context_lines:
            prompt += "[Context]\n" + "\n".join(context_lines) + "\n"

        prompt += (
            f"Produce only the {target_name} translation, without any additional explanations or commentary. "
            f"Please translate the following {source_name} text into {target_name}:\n\n\n"
            f"{text.strip()}<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )

        return prompt

    def _tokenize_inputs(self, text: str, source_lang: str, target_lang: str, context: TranslationContext | None):
        if context is not None:
            prompt = self._build_context_prompt(text, source_lang, target_lang, context)
            return self._processor.tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(
                self._model.device
            )

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
        return self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device)

    def translate(
        self, text: str, source_lang: str, target_lang: str, context: TranslationContext | None = None
    ) -> Iterator[str]:
        from transformers import TextIteratorStreamer

        inputs = self._tokenize_inputs(text, source_lang, target_lang, context)

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

    @property
    def model_name(self) -> str:
        return "FakeTranslator"

    def translate(
        self, text: str, source_lang: str, target_lang: str, context: TranslationContext | None = None
    ) -> Iterator[str]:
        target_name = self._languages.get(target_lang, target_lang)
        result = f"[{target_name}] {text}"
        accumulated = ""
        for char in result:
            accumulated += char
            yield accumulated
