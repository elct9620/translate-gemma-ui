import logging
from collections.abc import Iterator
from dataclasses import dataclass
from threading import Thread
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)

SUPPORTED_LANGUAGES: dict[str, str] = {
    "en": "English",
    "zh-TW": "Chinese (Traditional)",
    "ja": "Japanese",
}


@dataclass(frozen=True)
class TranslationContext:
    previous: list[str]
    following: list[str]
    glossary_prompt: str = ""


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


class TranslateGemmaTranslator:
    def __init__(self, model_id: str = "google/translategemma-4b-it", token: str | None = None):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        logger.info("Loading model %s...", model_id)

        self._model_name = model_id
        self._processor = AutoProcessor.from_pretrained(model_id, token=token)
        self._languages = SUPPORTED_LANGUAGES

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

    def _build_context_prompt(self, text: str, source_lang: str, target_lang: str, context: TranslationContext) -> str:
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

        if context.glossary_prompt:
            prompt += context.glossary_prompt

        prompt += (
            f"Produce only the {target_name} translation, without any additional explanations or commentary. "
            f"Please translate the following {source_name} text into {target_name}:\n\n\n"
            f"{text.strip()}<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )

        return prompt

    def _tokenize_inputs(self, text: str, source_lang: str, target_lang: str, context: TranslationContext | None):
        effective_context = context if context is not None else TranslationContext(previous=[], following=[])
        prompt = self._build_context_prompt(text, source_lang, target_lang, effective_context)
        return self._processor.tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(self._model.device)

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
