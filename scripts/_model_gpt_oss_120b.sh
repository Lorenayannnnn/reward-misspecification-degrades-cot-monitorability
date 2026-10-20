#!/usr/bin/env bash
# openai/gpt-oss-120b base model profile (sourced by the *_gpt_oss_120b.sh wrappers)

export BASE_MODEL="${BASE_MODEL:-openai/gpt-oss-120b}"

export MAX_TOKEN_ID="${MAX_TOKEN_ID:-200018}"

export THINK_START="${THINK_START:-<|channel|>analysis<|message|>}"
export THINK_END="${THINK_END:-<|end|><|start|>assistant<|channel|>final<|message|>}"


echo "[profile] policy=$BASE_MODEL think=$THINK_START..$THINK_END"
