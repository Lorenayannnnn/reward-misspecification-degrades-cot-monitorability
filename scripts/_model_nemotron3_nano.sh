#!/usr/bin/env bash
# nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 base model profile (sourced by the *_nemotron3_nano.sh wrappers):
# the model, its <think> markers and its tokenizer's highest id.

export BASE_MODEL="${BASE_MODEL:-nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16}"

export THINK_START="${THINK_START:-<think>}"
export THINK_END="${THINK_END:-</think>}"

echo "[profile] policy=$BASE_MODEL think=$THINK_START..$THINK_END"
