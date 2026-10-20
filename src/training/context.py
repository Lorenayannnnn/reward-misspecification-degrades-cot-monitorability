"""
Create renderer, tokenizer and monitor backend.
"""

from __future__ import annotations

from omegaconf import DictConfig

from src.monitor.backends import build_monitor_backend


def setup_renderer(model_name: str):
    from tinker_cookbook import model_info
    from tinker_cookbook.renderers import get_renderer
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    tokenizer = get_tokenizer(model_name)
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    print(f"[setup_renderer] renderer '{renderer_name}' for {model_name}")
    renderer = get_renderer(renderer_name, tokenizer)
    return tokenizer, renderer


def decode_full_text(tokens: list[int], tokenizer) -> str:
    return tokenizer.decode(tokens, skip_special_tokens=False)


def setup_generation_and_backends(args: DictConfig):
    """
    Build ``(tokenizer, renderer, stop_sequences, monitor_backend)``; the backend is None
    without ``monitor_backend_type``.
    """
    tokenizer, renderer = setup_renderer(args.base_model)
    stop_sequences = renderer.get_stop_sequences()

    monitor_backend = None
    if args.monitor_backend_type:
        monitor_backend = build_monitor_backend({
            "monitor_backend_type": args.monitor_backend_type,
            "monitor_model_name": args.monitor_model_name,
            "seed": args.seed,
            "think_start": args.get("monitor_think_start_str"),
            "think_end": args.get("monitor_think_end_str"),
            "max_tokens": args.monitor_max_tokens,
        })
        if monitor_backend:
            print(f"Monitor: {args.monitor_backend_type} / {args.monitor_model_name}")

    return tokenizer, renderer, stop_sequences, monitor_backend
