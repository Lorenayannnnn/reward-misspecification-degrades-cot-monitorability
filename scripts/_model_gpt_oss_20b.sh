#!/usr/bin/env bash
# openai/gpt-oss-20b base model profile (sourced by the *_gpt_oss_20b.sh wrappers): the model, its CoT markers
# and its tokenizer's highest id. Everything else is shared and lives in config_files/train.yaml.

export BASE_MODEL="${BASE_MODEL:-openai/gpt-oss-20b}"

export MAX_TOKEN_ID="${MAX_TOKEN_ID:-200018}"

export THINK_START="${THINK_START:-<|channel|>analysis<|message|>}"
export THINK_END="${THINK_END:-<|end|><|start|>assistant<|channel|>final<|message|>}"

echo "[profile] policy=$BASE_MODEL think=$THINK_START..$THINK_END"
