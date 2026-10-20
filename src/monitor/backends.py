"""
Classes and functions for creating and prompting model backends (openai / tinker)".
"""

from __future__ import annotations

import logging
import os
import re
from abc import ABC, abstractmethod

from src.utils import parse_out_main_cot_output

logger = logging.getLogger(__name__)

_STRICT_ANSWER_RE = re.compile(r"ANSWER:\s*\**\s*(YES|NO)\b", re.IGNORECASE)
_REASON_RE = re.compile(r"REASON:\s*(.*?)\s*(?:\n\s*ANSWER:|\Z)", re.IGNORECASE | re.DOTALL)

# Retries are for format failures only; transport errors are already retried with backoff by
# the provider SDK (tinker: the base client plus tinker/lib's RetryHandler).
_DEFAULT_MAX_FORMAT_RETRIES = 3

# Per-request timeout (seconds) for one monitor sample; a stalled call would otherwise hang
# the whole evaluation.
_DEFAULT_REQUEST_TIMEOUT = 180.0


def _parse_prediction_reason(final_output: str) -> tuple[int | None, str]:
    """
    Parse (prediction, reason) from a REASON/ANSWER final output. prediction is None when no
    ANSWER line is present.
    """
    matches = list(_STRICT_ANSWER_RE.finditer(final_output))
    prediction = None if not matches else (1 if matches[-1].group(1).upper() == "YES" else 0)

    reason_matches = list(_REASON_RE.finditer(final_output))
    reason = reason_matches[-1].group(1).strip() if reason_matches else ""
    return prediction, reason


class BaseMonitorBackend(ABC):

    max_format_retries: int

    @abstractmethod
    def complete(self, user_message: str, attempt: int) -> tuple[str | None, str | None, str]:
        """
        Send one API request to the monitor backend and parse out reasoning and output.
        """
        ...

    def predict_with_reason(self, user_message: str) -> tuple[int, str, bool, str]:
        """
        Prompt the monitor model to predict hacking with reasoning. Returns (prediction, reason, truncated, raw) where prediction is 1 for YES,
        """
        content = ""
        reason = ""
        for attempt in range(self.max_format_retries):
            try:
                _cot, final_output, content = self.complete(user_message, attempt)
            except Exception as e:
                logger.warning(f"Monitor sample failed after the SDK's retries: {type(e).__name__}: {e}", exc_info=True)
                break
            if final_output is None:
                continue    # truncated before the final output began: no prediction to read
            prediction, reason = _parse_prediction_reason(final_output)
            if prediction is not None:
                return prediction, reason, False, content
        logger.warning(f"Monitor response did not match expected format after "
                       f"{self.max_format_retries} tries: {content!r}")
        return 0, reason, True, content


class OpenAIMonitorBackend(BaseMonitorBackend):

    def __init__(
        self,
        model_name: str,
        seed: int,
        max_tokens: int,
        max_format_retries: int = _DEFAULT_MAX_FORMAT_RETRIES,
        request_timeout: float = _DEFAULT_REQUEST_TIMEOUT,
    ):
        self.model_name = model_name
        self.seed = seed
        self.max_tokens = max_tokens
        self.request_timeout = request_timeout
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("No OpenAI API key for the monitor backend: set OPENAI_API_KEY in keys.sh.")
        self.max_format_retries = max_format_retries
        try:
            import openai  # noqa: F401  only checks that the package is installed
        except ImportError:
            raise ImportError("The openai backend needs the `openai` package: pip install openai") from None

    def complete(self, user_message: str, attempt: int) -> tuple[None, str, str]:
        """
        Run one chat completion. The chat API returns only the final message, so cot is None and
        final_output and raw are the same text.
        """
        import openai

        client = openai.OpenAI(timeout=self.request_timeout)
        reply = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": user_message}],
            # max_completion_tokens: newer models reject max_tokens.
            max_completion_tokens=self.max_tokens,
            seed=self.seed + attempt,   # a retry must differ from the previous attempt
        )
        content = reply.choices[0].message.content or ""
        return None, content, content


class TinkerMonitorBackend(BaseMonitorBackend):

    def __init__(
        self,
        model_name: str,
        max_tokens: int,
        seed: int,
        base_url: str | None = None,
        think_start: str | None = None,
        think_end: str | None = None,
        max_format_retries: int = _DEFAULT_MAX_FORMAT_RETRIES,
        request_timeout: float = _DEFAULT_REQUEST_TIMEOUT,
    ):
        import tinker
        from tinker_cookbook import model_info, renderers
        from tinker_cookbook.tokenizer_utils import get_tokenizer

        self.model_name = model_name
        self.max_tokens = max_tokens
        self.seed = seed
        self.request_timeout = request_timeout
        self.max_format_retries = max_format_retries
        if (think_start is None) != (think_end is None):
            raise ValueError(
                f"Set both monitor think markers or neither: start={think_start!r}, end={think_end!r}"
            )
        self.think_start = think_start
        self.think_end = think_end

        self._tokenizer = get_tokenizer(model_name)
        self._renderer = renderers.get_renderer(model_info.get_recommended_renderer_name(model_name),
                                                self._tokenizer)

        # The service client reads TINKER_API_KEY itself (set in keys.sh).
        base_url = base_url or os.environ.get("TINKER_BASE_URL")
        service_client = tinker.ServiceClient(base_url=base_url)
        self._sampling_client = service_client.create_sampling_client(base_model=model_name)

    def complete(self, user_message: str, attempt: int) -> tuple[str | None, str | None, str]:
        import tinker

        model_input = self._renderer.build_generation_prompt([{"role": "user", "content": user_message}])
        sampling_params = tinker.types.SamplingParams(
            max_tokens=self.max_tokens, stop=self._renderer.get_stop_sequences(), seed=self.seed + attempt,
        )
        result = self._sampling_client.sample(
            prompt=model_input, num_samples=1, sampling_params=sampling_params,
        ).result(timeout=self.request_timeout)
        tokens = list(result.sequences[0].tokens)
        content = self._tokenizer.decode(tokens, skip_special_tokens=False)
        cot = None if self.think_start is None else parse_out_main_cot_output(
            tokens, self._tokenizer, self.think_start, self.think_end)[0]
        return cot, self._final_output(tokens, content), content

    def _final_output(self, tokens: list[int], content: str) -> str | None:
        if self.think_start is None:
            return self._tokenizer.decode(tokens, skip_special_tokens=True)
        _cot, final_output, *_ = parse_out_main_cot_output(
            list(tokens), self._tokenizer, self.think_start, self.think_end
        )
        return final_output


def build_monitor_backend(cfg) -> BaseMonitorBackend | None:
    """
    Build a monitor backend from config. None when type or model name are unset.
    """
    backend_type = cfg.get("monitor_backend_type")
    model_name = cfg.get("monitor_model_name")
    if not backend_type or not model_name:
        return None
    if backend_type == "openai":
        if cfg.get("think_start") or cfg.get("think_end"):
            raise ValueError(
                "monitor think markers only apply to the tinker backend: the OpenAI chat API "
                "returns the final message text without reasoning tokens."
            )
        return OpenAIMonitorBackend(
            model_name=model_name,
            seed=cfg["seed"],
            max_tokens=cfg["max_tokens"],
        )
    elif backend_type == "tinker":
        return TinkerMonitorBackend(
            model_name=model_name,
            seed=cfg["seed"],
            max_tokens=cfg["max_tokens"],
            base_url=cfg.get("base_url"),
            think_start=cfg.get("think_start"),
            think_end=cfg.get("think_end"),
        )
    else:
        raise ValueError(
            f"Unknown backend type: {backend_type!r}. Supported: 'openai', 'tinker'. "
            "Subclass BaseMonitorBackend for custom backends."
        )
